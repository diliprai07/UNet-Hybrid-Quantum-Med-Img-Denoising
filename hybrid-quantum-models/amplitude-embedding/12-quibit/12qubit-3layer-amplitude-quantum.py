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
# PERCEPTUAL LOSS
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
# QUANTUM BOTTLENECK WITH AMPLITUDE EMBEDDING (12 QUBITS)
# ========================
class QuantumBottleneck(nn.Module):
    def __init__(self, in_channels=512, n_qubits=12, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_features = 2 ** n_qubits  # Amplitude embedding needs 2^n_qubits features (4096 for 12 qubits)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        
        # Map from in_channels to 2^n_qubits features for amplitude encoding
        # For 12 qubits, we need 4096 features, so we add intermediate layers
        self.fc_in = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_features)
        )
        
        # Add normalization to ensure valid amplitude state
        self.norm = nn.Softmax(dim=1)  # Ensures positive probabilities summing to 1

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # inputs shape: (batch_size, 2^n_qubits)
            # Normalize inputs to form valid amplitude vector
            # The inputs should represent amplitudes for basis states
            amplitude_vector = inputs
            
            # Ensure the amplitude vector is normalized (L2 norm = 1)
            # This is required for valid quantum state
            norm = torch.sqrt(torch.sum(amplitude_vector ** 2, dim=1, keepdim=True))
            amplitude_vector = amplitude_vector / (norm + 1e-8)
            
            # Apply amplitude embedding
            qml.AmplitudeEmbedding(
                features=amplitude_vector, 
                wires=range(n_qubits),
                normalize=False  # Already normalized above
            )
            
            # Variational layers - more layers for 12 qubits to increase expressivity
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            
            # Return expectations for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
        # Map back from n_qubits to original channels
        self.fc_out = nn.Sequential(
            nn.Linear(n_qubits, 256),
            nn.ReLU(),
            nn.Linear(256, in_channels)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Global pooling and project to amplitude features
        pooled = self.pool(x).view(B, C)
        amplitude_features = self.fc_in(pooled)
        
        # Apply softmax to ensure valid probability distribution for amplitude embedding
        amplitude_features = self.norm(amplitude_features)
        
        # Quantum layer expects (B, 2^n_qubits)
        q_out = self.q_layer(amplitude_features)  # Shape: (B, n_qubits)
        
        # Project back to original channels
        out = self.fc_out(q_out).view(B, C, 1, 1)
        out = out.expand(-1, -1, H, W)
        
        # Residual connection
        return x + self.act(out)

# ========================
# UNET GENERATOR
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
    def forward(self,x): return self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = UNetBlock(1,64)
        self.d2 = UNetBlock(64,128)
        self.d3 = UNetBlock(128,256)
        self.d4 = UNetBlock(256,512)
        # Quantum bottleneck with amplitude embedding (12 qubits -> 4096 features)
        self.qb = QuantumBottleneck(512, 12, 4)  # Increased layers for 12 qubits
        self.u1 = UNetBlock(512,256,down=False)
        self.u2 = UNetBlock(512,128,down=False)
        self.u3 = UNetBlock(256,64,down=False)
        self.final = nn.ConvTranspose2d(128,1,4,2,1)

    def forward(self,x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d4 = self.qb(d4)
        u1 = self.u1(d4)
        u2 = self.u2(torch.cat([u1,d3],1))
        u3 = self.u3(torch.cat([u2,d2],1))
        return torch.sigmoid(self.final(torch.cat([u3,d1],1)))

# ========================
# DISCRIMINATOR
# ========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,64,4,2,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,1,4,1,1)
        )
    def forward(self,x): return self.model(x)

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
    # Base directory for this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(base_dir, "train_split/train_split")
    val_path   = os.path.join(base_dir, "valid_train/valid_train")

    save_dir   = os.path.join(base_dir, "amplitude_quantum_ckpt_12qubits")  # Different name to compare
    result_dir = os.path.join(base_dir, "amplitude_quantum_results_12qubits")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    train_loader = DataLoader(DatasetLoader(train_path, transform, 25),
                             batch_size=8, shuffle=True)
    val_loader = DataLoader(DatasetLoader(val_path, val_transform, 25),
                           batch_size=8)

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    # Count parameters
    g_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    print(f"Generator parameters: {g_params:,}")
    
    # Print quantum bottleneck info
    print(f"Using 12 qubits with 2^{12} = {2**12:,} amplitude features")

    optG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5,0.999))
    optD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5,0.999))

    adv_loss = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    perceptual = VGGPerceptualLoss().to(device)

    metrics_file = os.path.join(save_dir, "metrics.json")
    ckpt_path = os.path.join(save_dir, "latest.pth")

    start_epoch, best_psnr = 0, 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        netG.load_state_dict(ckpt['netG'])
        netD.load_state_dict(ckpt['netD'])
        optG.load_state_dict(ckpt['optG'])
        optD.load_state_dict(ckpt['optD'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt['best_psnr']
        print(f"Resumed from epoch {start_epoch}")

    if os.path.exists(metrics_file):
        try:
            metrics = json.load(open(metrics_file))
            if "val_loss" not in metrics: metrics["val_loss"] = []
        except:
            metrics = {"epoch":[], "psnr":[], "ssim":[], "lossG":[], "lossD":[], "val_loss":[]}
    else:
        metrics = {"epoch":[], "psnr":[], "ssim":[], "lossG":[], "lossD":[], "val_loss":[]}

    print("Starting training with Amplitude Embedding (12 qubits)...")
    
    for epoch in range(start_epoch, 300):
        netG.train(); netD.train()
        total_g, total_d = 0,0

        for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            noisy, clean = noisy.to(device), clean.to(device)

            real = torch.ones_like(netD(clean)) * 0.9
            fake = torch.zeros_like(netD(clean))

            # ---- D ----
            optD.zero_grad()
            real_noisy = add_gaussian_noise(clean.detach())
            real_noisy.requires_grad_(True)

            fake_img = netG(noisy).detach()
            fake_noisy = add_gaussian_noise(fake_img)

            real_pred = netD(real_noisy)
            fake_pred = netD(fake_noisy)

            lossD = adv_loss(real_pred, real) + adv_loss(fake_pred, fake)
            lossD += r1_penalty(real_pred, real_noisy)
            lossD.backward()
            optD.step()

            # ---- G ----
            optG.zero_grad()
            fake_img = netG(noisy)
            fake_pred = netD(fake_img)

            lossG = adv_loss(fake_pred, real)
            lossG += 100 * l1(fake_img, clean)
            lossG += 10 * perceptual(fake_img, clean)
            lossG.backward()
            optG.step()

            total_g += lossG.item()
            total_d += lossD.item()

        # ---- VALIDATION ----
        netG.eval()
        psnr_total, ssim_total, val_loss_total = 0,0,0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                out = netG(noisy)
                psnr_total += calculate_psnr(out, clean)
                ssim_total += calculate_ssim(out, clean)
                val_loss_total += l1(out, clean).item()

        psnr = psnr_total / len(val_loader)
        ssim = ssim_total / len(val_loader)
        avg_v = val_loss_total / len(val_loader)
        avg_g = total_g/len(train_loader)
        avg_d = total_d/len(train_loader)
        print(f"Epoch {epoch+1} | PSNR {psnr:.2f} | SSIM {ssim:.4f} | ValLoss {avg_v:.4f} | LossG {avg_g:.2f} | LossD {avg_d:.2f}")

        metrics["epoch"].append(epoch+1)
        metrics["psnr"].append(psnr)
        metrics["ssim"].append(ssim)
        metrics["val_loss"].append(avg_v)
        metrics["lossG"].append(avg_g)
        metrics["lossD"].append(avg_d)
        json.dump(metrics, open(metrics_file,"w"), indent=2)

        # SAVE CHECKPOINT
        torch.save({
            'epoch': epoch,
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'optG': optG.state_dict(),
            'optD': optD.state_dict(),
            'best_psnr': best_psnr
        }, ckpt_path)

        # BEST MODEL
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(netG.state_dict(), os.path.join(save_dir,"best.pth"))
            print(f"New best PSNR: {best_psnr:.2f}")

        # SAMPLE
        if (epoch+1) % 5 == 0:
            noisy, clean = next(iter(val_loader))
            noisy, clean = noisy.to(device), clean.to(device)
            out = netG(noisy)
            save_sample(noisy[0], out[0], clean[0],
                        os.path.join(result_dir,f"epoch_{epoch+1}.png"))

    print(f"Training completed! Best PSNR: {best_psnr:.2f}")

# ========================
# RUN
# ========================
if __name__ == "__main__":
    train()