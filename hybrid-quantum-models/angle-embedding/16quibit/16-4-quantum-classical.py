# ==============================
# QUANTUM U-NET DCGAN FOR DENOISING (Improved for Detail Preservation)
# ==============================
import os, json
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

# ========================
# CONFIGURATION
# ========================
USE_PERCEPTUAL = False      # Set to True to enable VGG perceptual loss (may cause smoothing)
BATCH_SIZE = 16
LAMBDA_ADV = 2.0            # Adversarial loss weight (increased)
LAMBDA_L1 = 5               # L1 loss weight (reduced)
LAMBDA_PER = 2              # Perceptual loss weight (reduced, only if used)
LAMBDA_R1 = 0.5             # R1 penalty weight
LR_G = 1e-4                 # Generator learning rate
LR_D = 2e-4                 # Discriminator learning rate
NUM_EPOCHS = 300
G_UPDATES_PER_D = 2         # Generator updates per discriminator update

# ========================
# METRICS
# ========================
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 100 if mse == 0 else 20 * log10(1.0 / torch.sqrt(mse).item())

def calculate_ssim(img1, img2):
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(img1, 11, 1, 5)
    mu2 = F.avg_pool2d(img2, 11, 1, 5)
    sigma1 = F.avg_pool2d(img1*img1, 11,1,5) - mu1**2
    sigma2 = F.avg_pool2d(img2*img2, 11,1,5) - mu2**2
    sigma12 = F.avg_pool2d(img1*img2, 11,1,5) - mu1*mu2
    ssim = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1)*(sigma1 + sigma2 + C2))
    return ssim.mean().item()

# ========================
# NOISE
# ========================
def add_awgn_noise(img, sigma=25):
    noise = torch.randn_like(img) * (sigma/255.0)
    return torch.clamp(img + noise, 0, 1)

def add_gaussian_noise(img, std=0.01):
    if not img.requires_grad:
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, 0, 1)
    return img

# ========================
# DATASET
# ========================
class DatasetLoader(Dataset):
    def __init__(self, root, transform=None, sigma=25):
        if not os.path.exists(root):
            raise FileNotFoundError(f"Directory {root} not found. Ensure you are running from the correct location.")
        self.files = [f for f in os.listdir(root)
                      if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
        self.root = root
        self.transform = transform
        self.sigma = sigma
        print(f"{len(self.files)} images loaded from {root}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        img = Image.open(img_path).convert('L')
        img = self.transform(img)
        noisy = add_awgn_noise(img, self.sigma)
        return noisy, img

# ========================
# PERCEPTUAL LOSS (optional)
# ========================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        x = x.repeat(1,3,1,1)
        y = y.repeat(1,3,1,1)
        return F.l1_loss(self.vgg(x), self.vgg(y))

# ========================
# QUANTUM BOTTLENECK (with local pooling)
# ========================
class QuantumBottleneck(nn.Module):
    def __init__(self, in_channels=512, n_qubits=16, n_layers=4, pool_size=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.pool_size = pool_size
        # Use a fixed average pooling to preserve local structure
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        # Input to quantum circuit: pool_size*pool_size*in_channels -> flatten to vector
        self.fc_in = nn.Linear(pool_size * pool_size * in_channels, n_qubits)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        self.fc_out = nn.Linear(n_qubits, pool_size * pool_size * in_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape
        # Pool to local patches
        pooled = self.pool(x)                       # (B, C, pool_size, pool_size)
        flat = pooled.view(B, -1)                   # (B, C*pool_size*pool_size)
        q_in = self.fc_in(flat)                     # (B, n_qubits)
        q_out = self.q_layer(q_in)                  # (B, n_qubits)
        out_flat = self.fc_out(q_out)               # (B, C*pool_size*pool_size)
        out = out_flat.view(B, C, self.pool_size, self.pool_size)
        # Upsample to original size using bilinear interpolation
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return x + self.act(out)

# ========================
# UNET GENERATOR (no normalization, shallower)
# ========================
class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.LeakyReLU(0.2)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.ReLU()
            )
    def forward(self,x): return self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Only 3 downsampling steps (64 -> 128 -> 256)
        self.d1 = UNetBlock(1, 64)
        self.d2 = UNetBlock(64, 128)
        self.d3 = UNetBlock(128, 256)
        # Quantum bottleneck at 16x16 (if input is 128x128, after 3 downsamplings: 128/2^3 = 16)
        self.qb = QuantumBottleneck(256, n_qubits=16, n_layers=4, pool_size=4)
        # Upsampling
        self.u1 = UNetBlock(256, 128, down=False)
        self.u2 = UNetBlock(256, 64, down=False)   # concatenation with d2 gives 128+128=256
        self.final = nn.ConvTranspose2d(128, 1, 4, 2, 1)  # concatenation with d1 gives 64+64=128

    def forward(self, x):
        d1 = self.d1(x)                     # 64x64
        d2 = self.d2(d1)                    # 32x32
        d3 = self.d3(d2)                    # 16x16
        d3 = self.qb(d3)                    # quantum enhanced
        u1 = self.u1(d3)                    # 32x32
        u2 = self.u2(torch.cat([u1, d2], 1)) # 64x64
        out = self.final(torch.cat([u2, d1], 1))
        return torch.sigmoid(out)

# ========================
# DISCRIMINATOR (with spectral normalization)
# ========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 3, 1, 1)   # patch output
        )
    def forward(self, x):
        return self.model(x)

# ========================
# SAVE SAMPLE
# ========================
def save_sample(n, d, c, path):
    plt.figure(figsize=(10,3))
    for i, img in enumerate([n,d,c]):
        plt.subplot(1,3,i+1)
        plt.imshow(img.detach().cpu().squeeze().numpy(), cmap='gray')
        plt.axis('off')
    plt.savefig(path)
    plt.close()

# ========================
# GRADIENT PENALTY
# ========================
def r1_penalty(real_pred, real_img, lambda_gp=1.0):
    B = real_pred.shape[0]
    mean_pred = real_pred.view(B, -1).mean(dim=1)
    gradients = grad(
        outputs=mean_pred,
        inputs=real_img,
        grad_outputs=torch.ones_like(mean_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_norm = gradients.view(B, -1).norm(2, dim=1)
    return lambda_gp * (grad_norm ** 2).mean()

# ========================
# TRAIN
# ========================
def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.abspath(os.path.join(base_dir, "..", "..", "Dataset", "train_split"))
    val_path   = os.path.abspath(os.path.join(base_dir, "..", "..", "Dataset", "valid_train"))
    save_dir   = os.path.join(base_dir, "16quantum_4qubit_ckpt")
    result_dir = os.path.join(base_dir, "16quantum_4qubit_results")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data augmentation
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    train_loader = DataLoader(DatasetLoader(train_path, transform, 25),
                             batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(DatasetLoader(val_path, val_transform, 25),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    optG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5,0.999))
    optD = optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5,0.999))

    adv_loss = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    if USE_PERCEPTUAL:
        perceptual = VGGPerceptualLoss().to(device)
    else:
        perceptual = None

    # Schedulers
    schedulerG = optim.lr_scheduler.CosineAnnealingLR(optG, T_max=NUM_EPOCHS)
    schedulerD = optim.lr_scheduler.CosineAnnealingLR(optD, T_max=NUM_EPOCHS)

    metrics_file = os.path.join(save_dir, "metrics.json")
    ckpt_path = os.path.join(save_dir, "latest.pth")

    start_epoch, best_psnr = 0, 0

    # Checkpoint loading with error handling
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            netG.load_state_dict(ckpt['netG'])
            netD.load_state_dict(ckpt['netD'])
            optG.load_state_dict(ckpt['optG'])
            optD.load_state_dict(ckpt['optD'])
            start_epoch = ckpt['epoch'] + 1
            best_psnr = ckpt['best_psnr']
            print(f"Resumed from epoch {start_epoch}")
        except RuntimeError as e:
            print(f"Checkpoint loading failed: {e}")
            print("Starting training from scratch (checkpoint will be overwritten).")
            backup_path = ckpt_path + ".old"
            if not os.path.exists(backup_path):
                os.rename(ckpt_path, backup_path)
                print(f"Moved old checkpoint to {backup_path}")
            else:
                os.remove(ckpt_path)
                print(f"Removed incompatible checkpoint.")

    # Load metrics
    if os.path.exists(metrics_file):
        try:
            metrics = json.load(open(metrics_file))
        except:
            metrics = {"epoch":[], "psnr":[], "ssim":[], "lossG":[], "lossD":[]}
    else:
        metrics = {"epoch":[], "psnr":[], "ssim":[], "lossG":[], "lossD":[]}

    for epoch in range(start_epoch, NUM_EPOCHS):
        netG.train()
        netD.train()
        total_g, total_d = 0, 0

        for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            noisy, clean = noisy.to(device), clean.to(device)

            real = torch.ones_like(netD(clean)) * 0.9
            fake = torch.zeros_like(netD(clean))

            # ---- Train Discriminator (once) ----
            optD.zero_grad()
            real_noisy = add_gaussian_noise(clean.detach())
            real_noisy.requires_grad_(True)

            fake_img = netG(noisy).detach()
            fake_noisy = add_gaussian_noise(fake_img)

            real_pred = netD(real_noisy)
            fake_pred = netD(fake_noisy)

            lossD = adv_loss(real_pred, real) + adv_loss(fake_pred, fake)
            lossD += LAMBDA_R1 * r1_penalty(real_pred, real_noisy)
            lossD.backward()
            optD.step()
            total_d += lossD.item()

            # ---- Train Generator (multiple times) ----
            for _ in range(G_UPDATES_PER_D):
                optG.zero_grad()
                fake_img = netG(noisy)
                fake_pred = netD(fake_img)

                lossG = LAMBDA_ADV * adv_loss(fake_pred, real)
                lossG += LAMBDA_L1 * l1(fake_img, clean)
                if USE_PERCEPTUAL:
                    lossG += LAMBDA_PER * perceptual(fake_img, clean)

                lossG.backward()
                optG.step()
                total_g += lossG.item()

        # --- Validation ---
        netG.eval()
        psnr_total, ssim_total = 0, 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                out = netG(noisy)
                psnr_total += calculate_psnr(out, clean)
                ssim_total += calculate_ssim(out, clean)

        psnr = psnr_total / len(val_loader)
        ssim = ssim_total / len(val_loader)
        print(f"Epoch {epoch+1} | PSNR {psnr:.2f} | SSIM {ssim:.4f}")

        # Record metrics
        metrics["epoch"].append(epoch+1)
        metrics["psnr"].append(psnr)
        metrics["ssim"].append(ssim)
        metrics["lossG"].append(total_g / (len(train_loader) * G_UPDATES_PER_D))
        metrics["lossD"].append(total_d / len(train_loader))
        json.dump(metrics, open(metrics_file, "w"), indent=2)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'optG': optG.state_dict(),
            'optD': optD.state_dict(),
            'best_psnr': best_psnr
        }, ckpt_path)

        # Save best model
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(netG.state_dict(), os.path.join(save_dir, "best.pth"))

        # Save sample images
        if (epoch+1) % 5 == 0:
            noisy, clean = next(iter(val_loader))
            noisy, clean = noisy.to(device), clean.to(device)
            out = netG(noisy)
            save_sample(noisy[0], out[0], clean[0],
                        os.path.join(result_dir, f"epoch_{epoch+1}.png"))

        # Update learning rates
        schedulerG.step()
        schedulerD.step()

if __name__ == "__main__":
    train()