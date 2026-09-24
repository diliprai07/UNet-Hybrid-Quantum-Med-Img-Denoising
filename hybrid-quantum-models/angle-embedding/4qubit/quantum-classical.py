# ==============================
# U‑NET DCGAN FOR DENOISING
# WITH OPTIONAL QUANTUM BOTTLENECK
# FIXED OVERSMOOTHING + FIXED NOISE (σ=25)
# ==============================

import os, json, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from math import log10
from torch.autograd import grad
import pennylane as qml
import numpy as np

# ========================
# CONFIGURATION
# ========================
USE_QUANTUM      = True          # Set False for non‑quantum baseline
BATCH_SIZE       = 16
LAMBDA_ADV       = 1.0
LAMBDA_L1        = 5.0           # REDUCED from 10 to reduce blur
LAMBDA_FREQ      = 1.0           # INCREASED from 0.2 to preserve high freqs
LAMBDA_EDGE      = 2.0           # INCREASED from 0.5 to sharpen edges
LAMBDA_SSIM      = 0.5           # INCREASED from 0.2
LAMBDA_LAPLACE   = 0.5           # Laplacian loss to penalise blur
LAMBDA_PER       = 1.0
LAMBDA_R1        = 0.1
LR_G             = 1e-4
LR_D             = 2e-4
NUM_EPOCHS       = 300
G_UPDATES_PER_D  = 2

# Noise parameters (fixed sigma = 25 for both training and validation)
NOISE_SIGMA      = 25            # standard deviation (0-255 range)

# Data augmentation
CROP_SIZE        = 128
USE_RANDOM_CROP  = False         # set True if images > 128

# EMA
EMA_DECAY        = 0.999

# ========================
# METRICS
# ========================
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 100.0 if mse == 0 else 20 * log10(1.0 / torch.sqrt(mse).item())

def calculate_ssim(img1, img2):
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(img1, 11, 1, 5)
    mu2 = F.avg_pool2d(img2, 11, 1, 5)
    sigma1 = F.avg_pool2d(img1*img1, 11,1,5) - mu1**2
    sigma2 = F.avg_pool2d(img2*img2, 11,1,5) - mu2**2
    sigma12 = F.avg_pool2d(img1*img2, 11,1,5) - mu1*mu2
    ssim = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1)*(sigma1 + sigma2 + C2))
    return ssim.mean().item()

# ========================
# LOSS FUNCTIONS (anti‑oversmoothing)
# ========================
def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target)**2 + eps**2))

def frequency_loss(fake, clean):
    fft_fake  = torch.fft.fft2(fake)
    fft_clean = torch.fft.fft2(clean)
    return torch.mean(torch.abs(fft_fake - fft_clean))

def sobel_edge(x):
    kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],
                      dtype=torch.float32).view(1,1,3,3).to(x.device)
    ky = kx.transpose(2,3)
    B,C,H,W = x.shape
    xf = x.view(B*C,1,H,W)
    mag = torch.sqrt(F.conv2d(xf,kx,padding=1)**2 +
                     F.conv2d(xf,ky,padding=1)**2 + 1e-6)
    return mag.view(B,C,H,W)

def edge_loss(fake, clean):
    return F.l1_loss(sobel_edge(fake), sobel_edge(clean))

def ssim_loss(fake, clean):
    C1,C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(fake, 11,1,5)
    mu2 = F.avg_pool2d(clean, 11,1,5)
    s1 = F.avg_pool2d(fake*fake, 11,1,5) - mu1**2
    s2 = F.avg_pool2d(clean*clean, 11,1,5) - mu2**2
    s12 = F.avg_pool2d(fake*clean, 11,1,5) - mu1*mu2
    ssim_map = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))
    return 1.0 - ssim_map.mean()

def laplacian_loss(fake, clean):
    """
    Penalise differences in high-frequency components using a Laplacian filter.
    This encourages sharpness.
    """
    kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                          dtype=torch.float32).view(1,1,3,3).to(fake.device)
    fake_lap = F.conv2d(fake, kernel, padding=1)
    clean_lap = F.conv2d(clean, kernel, padding=1)
    return F.l1_loss(fake_lap, clean_lap)

# ========================
# NOISE FUNCTION (fixed AWGN)
# ========================
def add_awgn_noise(img, sigma):
    """Add additive white Gaussian noise with standard deviation sigma (0-255 range)."""
    noise = torch.randn_like(img) * (sigma / 255.0)
    return torch.clamp(img + noise, 0.0, 1.0)

# ========================
# DATASET
# ========================
class DatasetLoader(Dataset):
    def __init__(self, root, transform=None, noise_sigma=NOISE_SIGMA):
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        self.files = [f for f in os.listdir(root)
                      if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in: {root}")
        self.root = root
        self.transform = transform
        self.noise_sigma = noise_sigma
        print(f"{len(self.files)} images loaded from {root} (noise sigma={noise_sigma})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx])).convert('L')
        if self.transform:
            img = self.transform(img)          # (1, H, W) after ToTensor

        _, H, W = img.shape
        if USE_RANDOM_CROP and H > CROP_SIZE and W > CROP_SIZE:
            top = random.randint(0, H - CROP_SIZE)
            left = random.randint(0, W - CROP_SIZE)
            img = img[:, top:top+CROP_SIZE, left:left+CROP_SIZE]
        elif H != CROP_SIZE or W != CROP_SIZE:
            img = F.interpolate(img.unsqueeze(0), size=(CROP_SIZE, CROP_SIZE),
                                mode='bilinear', align_corners=False).squeeze(0)

        # Add fixed AWGN
        noisy = add_awgn_noise(img, self.noise_sigma)
        return noisy, img

# ========================
# VGG PERCEPTUAL LOSS
# ========================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        return F.l1_loss(self.vgg(x), self.vgg(y))

# ========================
# QUANTUM BOTTLENECK (small, stable)
# ========================
class QuantumBottleneck(nn.Module):
    def __init__(self, in_channels=512, n_qubits=4, n_layers=1, pool_size=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.pool_size = pool_size
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fc_in = nn.Linear(pool_size*pool_size*in_channels, n_qubits)

        dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.q_layer = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 3)})
        self.fc_out = nn.Linear(n_qubits, pool_size*pool_size*in_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        B,C,H,W = x.shape
        pooled = self.pool(x).view(B, -1)
        q_in = self.fc_in(pooled)
        q_out = self.q_layer(q_in)
        out = self.fc_out(q_out).view(B,C,self.pool_size,self.pool_size)
        out = F.interpolate(out, size=(H,W), mode='bilinear', align_corners=False)
        return x + self.act(out)

# ========================
# GENERATOR (with optional quantum bottleneck)
# ========================
class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )
    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, use_quantum=True):
        super().__init__()
        self.use_quantum = use_quantum
        self.d1 = UNetBlock(1, 64)
        self.d2 = UNetBlock(64, 128)
        self.d3 = UNetBlock(128, 256)
        self.d4 = UNetBlock(256, 512)

        if self.use_quantum:
            self.qb = QuantumBottleneck(512, n_qubits=4, n_layers=1, pool_size=2)
        else:
            self.qb = nn.Identity()   # placeholder

        self.u1 = UNetBlock(512, 256, down=False)
        self.u2 = UNetBlock(512, 128, down=False)
        self.u3 = UNetBlock(256, 64, down=False)

        self.final = nn.ConvTranspose2d(128, 1, 4, 2, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        d4 = self.qb(d4)                     # quantum or identity

        u1 = self.u1(d4)
        u2 = self.u2(torch.cat([u1, d3], 1))
        u3 = self.u3(torch.cat([u2, d2], 1))

        noise_pred = torch.tanh(self.final(torch.cat([u3, d1], 1)))
        out = torch.clamp(x - noise_pred, 0.0, 1.0)
        return out, noise_pred

# ========================
# MULTI‑SCALE DISCRIMINATOR (with spectral norm)
# ========================
class SingleScaleD(nn.Module):
    def __init__(self):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.model = nn.Sequential(
            sn(nn.Conv2d(1, 64, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(256, 512, 4, 2, 1)), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 3, 1, 1)
        )
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.D_fine = SingleScaleD()
        self.D_coarse = SingleScaleD()
        self.down = nn.AvgPool2d(2)

    def forward(self, x):
        return self.D_fine(x), self.D_coarse(self.down(x))

# ========================
# EMA
# ========================
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
    def update(self):
        sd = self.model.state_dict()
        for k in self.shadow:
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * sd[k].float()
    def apply_shadow(self):
        self.model.load_state_dict({k: v.to(next(self.model.parameters()).device)
                                    for k, v in self.shadow.items()})
    def restore(self, backup):
        self.model.load_state_dict(backup)

# ========================
# HELPER FUNCTIONS
# ========================
adv_criterion = nn.BCEWithLogitsLoss()

def adv_loss_multiscale(preds, label_val):
    if isinstance(preds, tuple):
        return sum(adv_criterion(p, torch.full_like(p, label_val)) for p in preds)
    return adv_criterion(preds, torch.full_like(preds, label_val))

def r1_penalty(real_pred, real_img, lambda_gp=1.0):
    pred = real_pred[0] if isinstance(real_pred, tuple) else real_pred
    B = pred.shape[0]
    mean_pred = pred.view(B, -1).mean(dim=1)
    g = grad(outputs=mean_pred, inputs=real_img,
             grad_outputs=torch.ones_like(mean_pred),
             create_graph=True, retain_graph=True, only_inputs=True)[0]
    return lambda_gp * (g.view(B, -1).norm(2, dim=1) ** 2).mean()

def save_sample(n, d, c, path):
    plt.figure(figsize=(10, 3))
    titles = ["Noisy", "Denoised", "Clean"]
    for i, img in enumerate([n, d, c]):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(img.detach().cpu().squeeze().numpy(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ========================
# TRAINING LOOP
# ========================
def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.abspath(os.path.join(base_dir, "..", "..", "..", "Dataset", "train_split"))
    val_path   = os.path.abspath(os.path.join(base_dir, "..", "..", "..", "Dataset", "valid_train"))
    suffix = "quantum" if USE_QUANTUM else "baseline"
    save_dir   = os.path.join(base_dir, f"quantum_classicalv5{suffix}")
    result_dir = os.path.join(base_dir, f"quantum_classical_v5_results_{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Quantum: {USE_QUANTUM} | Noise sigma: {NOISE_SIGMA}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create datasets with fixed noise sigma (no curriculum)
    train_dataset = DatasetLoader(train_path, train_transform, noise_sigma=NOISE_SIGMA)
    val_dataset   = DatasetLoader(val_path, val_transform, noise_sigma=NOISE_SIGMA)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(device=="cuda"))

    netG = Generator(use_quantum=USE_QUANTUM).to(device)
    netD = Discriminator().to(device)

    optG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.999))

    schedulerG = optim.lr_scheduler.CosineAnnealingWarmRestarts(optG, T_0=30, T_mult=2)
    schedulerD = optim.lr_scheduler.CosineAnnealingWarmRestarts(optD, T_0=30, T_mult=2)

    perceptual = VGGPerceptualLoss().to(device)
    ema = EMA(netG, decay=EMA_DECAY)

    metrics_file = os.path.join(save_dir, "metrics.json")
    ckpt_path = os.path.join(save_dir, "latest.pth")
    start_epoch, best_psnr = 0, 0.0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        netG.load_state_dict(ckpt['netG'])
        netD.load_state_dict(ckpt['netD'])
        optG.load_state_dict(ckpt['optG'])
        optD.load_state_dict(ckpt['optD'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt['best_psnr']
        if 'ema_shadow' in ckpt:
            ema.shadow = {k: v.to(device) for k, v in ckpt['ema_shadow'].items()}
        print(f"Resumed from epoch {start_epoch}, best PSNR {best_psnr:.2f}")

    metrics = {}
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
    else:
        metrics = {"epoch":[], "psnr":[], "ssim":[], "val_loss":[],
                   "lossG":[], "lossD":[], "loss_freq":[], "loss_edge":[],
                   "loss_ssim":[], "loss_laplace":[]}

    for epoch in range(start_epoch, NUM_EPOCHS):
        netG.train()
        netD.train()
        total_g = total_d = total_freq = total_edge = total_ssim_l = total_laplace = 0

        for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            noisy, clean = noisy.to(device), clean.to(device)

            real_label, fake_label = 0.9, 0.0

            # ------------------ Discriminator ------------------
            optD.zero_grad()

            real_for_r1 = clean.detach().clone().requires_grad_(True)
            r1_preds = netD(real_for_r1)
            r1 = LAMBDA_R1 * r1_penalty(r1_preds, real_for_r1)

            real_aug = add_awgn_noise(clean.detach(), sigma=2.5)   # tiny noise
            fake_img, _ = netG(noisy)
            fake_aug = fake_img.detach()

            real_preds = netD(real_aug)
            fake_preds = netD(fake_aug)

            lossD = adv_loss_multiscale(real_preds, real_label) + \
                    adv_loss_multiscale(fake_preds, fake_label) + r1
            lossD.backward()
            optD.step()
            total_d += lossD.item()

            # ------------------ Generator (multiple updates) ------------------
            for _ in range(G_UPDATES_PER_D):
                optG.zero_grad()

                fake_img, _ = netG(noisy)
                fake_preds = netD(fake_img)

                lossG_adv = LAMBDA_ADV * adv_loss_multiscale(fake_preds, real_label)
                lossG_l1 = LAMBDA_L1 * charbonnier_loss(fake_img, clean)

                lfreq = frequency_loss(fake_img, clean)
                ledge = edge_loss(fake_img, clean)
                lssim = ssim_loss(fake_img, clean)
                llaplace = laplacian_loss(fake_img, clean)

                lossG = lossG_adv + lossG_l1 + \
                        LAMBDA_FREQ*lfreq + LAMBDA_EDGE*ledge + \
                        LAMBDA_SSIM*lssim + LAMBDA_LAPLACE*llaplace

                if perceptual is not None:
                    lossG += LAMBDA_PER * perceptual(fake_img, clean)

                lossG.backward()
                optG.step()
                ema.update()

                total_g      += lossG.item()
                total_freq   += lfreq.item()
                total_edge   += ledge.item()
                total_ssim_l += lssim.item()
                total_laplace += llaplace.item()

        # ------------------ Validation (with EMA) ------------------
        train_backup = {k: v.clone() for k, v in netG.state_dict().items()}
        ema.apply_shadow()
        netG.eval()
        psnr_total = ssim_total = val_loss_total = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                out, _ = netG(noisy)
                psnr_total += calculate_psnr(out, clean)
                ssim_total += calculate_ssim(out, clean)
                val_loss_total += charbonnier_loss(out, clean).item()
        ema.restore(train_backup)
        netG.train()

        n_val = len(val_loader)
        psnr = psnr_total / n_val
        ssim = ssim_total / n_val
        val_loss = val_loss_total / n_val

        print(f"Epoch {epoch+1} | PSNR {psnr:.2f} dB | SSIM {ssim:.4f} | Val Loss {val_loss:.4f} | "
              f"G loss {total_g/(len(train_loader)*G_UPDATES_PER_D):.4f} | "
              f"D loss {total_d/len(train_loader):.4f} | "
              f"Laplace {total_laplace/(len(train_loader)*G_UPDATES_PER_D):.4f}")

        # Save metrics
        metrics["epoch"].append(epoch+1)
        metrics["psnr"].append(psnr)
        metrics["ssim"].append(ssim)
        metrics["val_loss"].append(val_loss)
        metrics["lossG"].append(total_g/(len(train_loader)*G_UPDATES_PER_D))
        metrics["lossD"].append(total_d/len(train_loader))
        metrics["loss_freq"].append(total_freq/(len(train_loader)*G_UPDATES_PER_D))
        metrics["loss_edge"].append(total_edge/(len(train_loader)*G_UPDATES_PER_D))
        metrics["loss_ssim"].append(total_ssim_l/(len(train_loader)*G_UPDATES_PER_D))
        metrics["loss_laplace"].append(total_laplace/(len(train_loader)*G_UPDATES_PER_D))
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'optG': optG.state_dict(),
            'optD': optD.state_dict(),
            'best_psnr': best_psnr,
            'ema_shadow': {k: v.cpu() for k, v in ema.shadow.items()}
        }, ckpt_path)

        # Save best model (EMA weights)
        if psnr > best_psnr:
            best_psnr = psnr
            ema.apply_shadow()
            torch.save(netG.state_dict(), os.path.join(save_dir, "best.pth"))
            ema.restore(train_backup)
            print(f" ★ New best PSNR: {best_psnr:.2f} dB")

        # Sample images every 5 epochs
        if (epoch+1) % 5 == 0:
            noisy, clean = next(iter(val_loader))
            noisy, clean = noisy.to(device), clean.to(device)
            ema.apply_shadow()
            netG.eval()
            with torch.no_grad():
                out, _ = netG(noisy)
            ema.restore(train_backup)
            netG.train()
            save_sample(noisy[0], out[0], clean[0], os.path.join(result_dir, f"epoch_{epoch+1}.png"))

        schedulerG.step()
        schedulerD.step()

if __name__ == "__main__":
    train()