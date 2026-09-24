import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from math import log10
import pennylane as qml
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
import piq
import numpy as np
import bm3d
from skimage.restoration import denoise_tv_chambolle

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

def calculate_mse(img1, img2):
    return F.mse_loss(img1, img2).item()

def calculate_fsim(img1, img2):
    # FSIM works directly with grayscale
    return piq.fsim(img1, img2, reduction='mean').item()

def prepare_for_fid(img):
    """Convert [0, 1] grayscale tensor to [0, 255] RGB uint8 tensor."""
    img_rgb = img.repeat(1, 3, 1, 1)
    img_uint8 = (img_rgb * 255).clamp(0, 255).to(torch.uint8)
    return img_uint8

# ========================
# NOISE
# ========================
def add_awgn_noise(img, sigma=25):
    noise = torch.randn_like(img) * (sigma/255.0)
    return torch.clamp(img + noise, 0, 1)

# ========================
# DATASET
# ========================
class TestDatasetLoader(Dataset):
    def __init__(self, root, transform=None, sigma=25):
        if not os.path.exists(root):
            raise FileNotFoundError(f"Directory {root} not found.")
        self.files = [f for f in os.listdir(root)
                      if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
        self.root = root
        self.transform = transform
        self.sigma = sigma

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        img = Image.open(img_path).convert('L')
        img = self.transform(img)
        noisy = add_awgn_noise(img, self.sigma)
        return noisy, img

# ========================
# ARCHITECTURES
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
    def forward(self,x): 
        return self.block(x)

# 4-qubit Quantum Bottleneck with Amplitude Embedding
class QuantumBottleneckAmp4(nn.Module):
    def __init__(self, in_channels=512, n_qubits=4, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_features = 2 ** n_qubits
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_in = nn.Linear(in_channels, self.n_features)
        self.norm = nn.Softmax(dim=1)
        
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            amplitude_vector = inputs
            norm = torch.sqrt(torch.sum(amplitude_vector ** 2, dim=1, keepdim=True))
            amplitude_vector = amplitude_vector / (norm + 1e-8)
            qml.AmplitudeEmbedding(features=amplitude_vector, wires=range(n_qubits), normalize=False)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.fc_out = nn.Linear(n_qubits, in_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape
        pooled = self.pool(x).view(B, C)
        amplitude_features = self.norm(self.fc_in(pooled))
        q_out = self.q_layer(amplitude_features)
        out = self.fc_out(q_out).view(B, C, 1, 1).expand(-1, -1, H, W)
        return x + self.act(out)

# 4-qubit Amplitude Generator
class GeneratorQuantum4(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = UNetBlock(1,64)
        self.d2 = UNetBlock(64,128)
        self.d3 = UNetBlock(128,256)
        self.d4 = UNetBlock(256,512)
        self.qb = QuantumBottleneckAmp4(512, 4, 3)
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

# Classical U-Net Generator
class GeneratorClassical(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = UNetBlock(1,64)
        self.d2 = UNetBlock(64,128)
        self.d3 = UNetBlock(128,256)
        self.d4 = UNetBlock(256,512)
        self.u1 = UNetBlock(512,256,down=False)
        self.u2 = UNetBlock(512,128,down=False)
        self.u3 = UNetBlock(256,64,down=False)
        self.final = nn.ConvTranspose2d(128,1,4,2,1)

    def forward(self,x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        u1 = self.u1(d4)
        u2 = self.u2(torch.cat([u1,d3],1))
        u3 = self.u3(torch.cat([u2,d2],1))
        return torch.sigmoid(self.final(torch.cat([u3,d1],1)))

# Standard DnCNN Architecture
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return x - out

def get_dncnn_model(sigma, device):
    weights_path = r"c:\Users\Gurung\Desktop\drp\DCGAN\models\classical_weights\dncnn_25.pth"
    model = DnCNN().to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model

# ========================
# EVALUATION METHODS
# ========================
def evaluate_model(model, ckpt_path, loader, device, fid_metric, lpips_fn):
    print(f"Loading checkpoint from: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if 'netG' in ckpt:
            model.load_state_dict(ckpt['netG'])
        else:
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f"Error loading checkpoint {ckpt_path}: {e}")
        return None, None, None, None, None

    model.eval()
    fid_metric.reset()  # Reset FID metric for each model
    psnr_total, ssim_total, lpips_total, mse_total, fsim_total = 0.0, 0.0, 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for noisy, clean in tqdm(loader, desc=f"Denoising with model"):
            noisy, clean = noisy.to(device), clean.to(device)
            out = model(noisy)
            out = torch.clamp(out, 0.0, 1.0)
            
            for i in range(out.size(0)):
                psnr_total += calculate_psnr(out[i:i+1], clean[i:i+1])
                ssim_total += calculate_ssim(out[i:i+1], clean[i:i+1])
                mse_total += calculate_mse(out[i:i+1], clean[i:i+1])
                fsim_total += calculate_fsim(out[i:i+1], clean[i:i+1])
                
                o_scaled = out[i:i+1] * 2.0 - 1.0
                c_scaled = clean[i:i+1] * 2.0 - 1.0
                lpips_total += lpips_fn(o_scaled, c_scaled).item()
            
            fid_metric.update(prepare_for_fid(out).to(device), real=False)

    num_samples = len(loader.dataset)
    return (psnr_total/num_samples, ssim_total/num_samples, lpips_total/num_samples, 
            mse_total/num_samples, fsim_total/num_samples)

def evaluate_dncnn_model(model, loader, device, fid_metric, lpips_fn):
    psnr_total, ssim_total, lpips_total, mse_total, fsim_total = 0.0, 0.0, 0.0, 0.0, 0.0
    model.eval()
    fid_metric.reset()
    
    with torch.no_grad():
        for noisy, clean in tqdm(loader, desc="Denoising with DnCNN"):
            noisy, clean = noisy.to(device), clean.to(device)
            out = model(noisy)
            out = torch.clamp(out, 0.0, 1.0)
            
            for i in range(out.size(0)):
                psnr_total += calculate_psnr(out[i:i+1], clean[i:i+1])
                ssim_total += calculate_ssim(out[i:i+1], clean[i:i+1])
                mse_total += calculate_mse(out[i:i+1], clean[i:i+1])
                fsim_total += calculate_fsim(out[i:i+1], clean[i:i+1])
                
                o_scaled = out[i:i+1] * 2.0 - 1.0
                c_scaled = clean[i:i+1] * 2.0 - 1.0
                lpips_total += lpips_fn(o_scaled, c_scaled).item()
            
            fid_metric.update(prepare_for_fid(out).to(device), real=False)

    num_samples = len(loader.dataset)
    return (psnr_total/num_samples, ssim_total/num_samples, lpips_total/num_samples, 
            mse_total/num_samples, fsim_total/num_samples)

def evaluate_bm3d_model(loader, device, fid_metric, lpips_fn, sigma=25):
    psnr_total, ssim_total, lpips_total, mse_total, fsim_total = 0.0, 0.0, 0.0, 0.0, 0.0
    sigma_norm = sigma / 255.0
    fid_metric.reset()
    
    for noisy, clean in tqdm(loader, desc="Denoising with BM3D"):
        B = noisy.size(0)
        out_batch = []
        for i in range(B):
            nz_np = noisy[i].cpu().squeeze().numpy()
            out_np = bm3d.bm3d(nz_np, sigma_psd=sigma_norm, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            out_np = np.clip(out_np, 0.0, 1.0)
            out_t = torch.from_numpy(out_np).float().unsqueeze(0).unsqueeze(0).to(device)
            out_batch.append(out_t)
            
            gt_t = clean[i:i+1].to(device)
            psnr_total += calculate_psnr(out_t, gt_t)
            ssim_total += calculate_ssim(out_t, gt_t)
            mse_total += calculate_mse(out_t, gt_t)
            fsim_total += calculate_fsim(out_t, gt_t)
            
            o_scaled = out_t * 2.0 - 1.0
            c_scaled = gt_t * 2.0 - 1.0
            lpips_total += lpips_fn(o_scaled, c_scaled).item()
            
        out_batch_t = torch.cat(out_batch, dim=0)
        fid_metric.update(prepare_for_fid(out_batch_t).to(device), real=False)

    num_samples = len(loader.dataset)
    return (psnr_total/num_samples, ssim_total/num_samples, lpips_total/num_samples, 
            mse_total/num_samples, fsim_total/num_samples)

def evaluate_tv_model(loader, device, fid_metric, lpips_fn, sigma=25):
    psnr_total, ssim_total, lpips_total, mse_total, fsim_total = 0.0, 0.0, 0.0, 0.0, 0.0
    tv_weight = 0.10 if sigma == 25 else 0.05
    fid_metric.reset()
    
    for noisy, clean in tqdm(loader, desc="Denoising with TV Chambolle"):
        B = noisy.size(0)
        out_batch = []
        for i in range(B):
            nz_np = noisy[i].cpu().squeeze().numpy()
            out_np = denoise_tv_chambolle(nz_np, weight=tv_weight)
            out_np = np.clip(out_np, 0.0, 1.0)
            out_t = torch.from_numpy(out_np).float().unsqueeze(0).unsqueeze(0).to(device)
            out_batch.append(out_t)
            
            gt_t = clean[i:i+1].to(device)
            psnr_total += calculate_psnr(out_t, gt_t)
            ssim_total += calculate_ssim(out_t, gt_t)
            mse_total += calculate_mse(out_t, gt_t)
            fsim_total += calculate_fsim(out_t, gt_t)
            
            o_scaled = out_t * 2.0 - 1.0
            c_scaled = gt_t * 2.0 - 1.0
            lpips_total += lpips_fn(o_scaled, c_scaled).item()
            
        out_batch_t = torch.cat(out_batch, dim=0)
        fid_metric.update(prepare_for_fid(out_batch_t).to(device), real=False)

    num_samples = len(loader.dataset)
    return (psnr_total/num_samples, ssim_total/num_samples, lpips_total/num_samples, 
            mse_total/num_samples, fsim_total/num_samples)

# ========================
# INFERENCE TIME MEASUREMENT (UPDATED)
# ========================
def measure_inference_time_model(model, ckpt_path, loader, device, num_warmup=3):
    """Measure average inference time per image for PyTorch models."""
    print(f"  Loading checkpoint from: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if 'netG' in ckpt:
            model.load_state_dict(ckpt['netG'])
        else:
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f"  Error loading checkpoint {ckpt_path}: {e}")
        return None

    model.eval()
    total_time = 0.0
    total_images = 0

    # Clear GPU cache before timing
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        # Warmup runs (not timed)
        print(f"  Warming up model ({num_warmup} iterations)...")
        warmup_done = 0
        for noisy, _ in loader:
            noisy = noisy.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            _ = model(noisy)
            if device == "cuda":
                torch.cuda.synchronize()
            warmup_done += 1
            if warmup_done >= num_warmup:
                break

        # Timed runs
        for noisy, _ in tqdm(loader, desc="  Timing inference"):
            noisy = noisy.to(device)
            B = noisy.size(0)

            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(noisy)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            total_time += (end - start)
            total_images += B

    avg_time = total_time / total_images
    return avg_time

def measure_inference_time_dncnn(model, loader, device, num_warmup=3):
    """Measure average inference time per image for DnCNN."""
    model.eval()
    total_time = 0.0
    total_images = 0

    # Clear GPU cache before timing
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    with torch.no_grad():
        # Warmup runs
        print(f"  Warming up DnCNN ({num_warmup} iterations)...")
        warmup_done = 0
        for noisy, _ in loader:
            noisy = noisy.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            _ = model(noisy)
            if device == "cuda":
                torch.cuda.synchronize()
            warmup_done += 1
            if warmup_done >= num_warmup:
                break

        # Timed runs
        for noisy, _ in tqdm(loader, desc="  Timing DnCNN inference"):
            noisy = noisy.to(device)
            B = noisy.size(0)

            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(noisy)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            total_time += (end - start)
            total_images += B

    return total_time / total_images

def measure_inference_time_bm3d(loader, sigma=25, max_images=3, num_warmup=2):
    """Measure average inference time per image for BM3D (complete pipeline)."""
    sigma_norm = sigma / 255.0
    total_time = 0.0
    total_images = 0
    
    # Warmup - BM3D has JIT compilation on first call
    print("  Warming up BM3D denoising...")
    sample_noisy = None
    for noisy, _ in loader:
        sample_noisy = noisy[0].cpu().squeeze().numpy()
        break
    
    for i in range(num_warmup):
        _ = bm3d.bm3d(sample_noisy, sigma_psd=sigma_norm, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    
    # Timed runs with complete pipeline
    for noisy, _ in tqdm(loader, desc="  Timing BM3D inference"):
        B = noisy.size(0)
        for i in range(B):
            if max_images is not None and total_images >= max_images:
                break
            
            nz_np = noisy[i].cpu().squeeze().numpy()
            
            start = time.perf_counter()
            # Complete denoising pipeline (same as evaluate_bm3d_model)
            out_np = bm3d.bm3d(nz_np, sigma_psd=sigma_norm, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            out_np = np.clip(out_np, 0.0, 1.0)
            _ = torch.from_numpy(out_np).float().unsqueeze(0).unsqueeze(0)
            end = time.perf_counter()
            
            total_time += (end - start)
            total_images += 1
            
        if max_images is not None and total_images >= max_images:
            break
    
    # pyrefly: ignore [division-by-zero]
    return total_time / total_images if total_images > 0 else 0.0

def measure_inference_time_tv(loader, sigma=25, num_warmup=3):
    """Measure average inference time per image for TV Chambolle (complete pipeline)."""
    tv_weight = 0.10 if sigma == 25 else 0.05
    total_time = 0.0
    total_images = 0
    
    # Warmup - run a few iterations without timing
    print("  Warming up TV denoising...")
    sample_noisy = None
    for noisy, _ in loader:
        sample_noisy = noisy[0].cpu().squeeze().numpy()
        break
    
    for i in range(num_warmup):
        _ = denoise_tv_chambolle(sample_noisy, weight=tv_weight)
    
    # Timed runs with complete pipeline
    for noisy, _ in tqdm(loader, desc="  Timing TV inference"):
        B = noisy.size(0)
        for i in range(B):
            nz_np = noisy[i].cpu().squeeze().numpy()
            
            start = time.perf_counter()
            # Complete denoising pipeline (same as evaluate_tv_model)
            out_np = denoise_tv_chambolle(nz_np, weight=tv_weight)
            out_np = np.clip(out_np, 0.0, 1.0)
            _ = torch.from_numpy(out_np).float().unsqueeze(0).unsqueeze(0)
            end = time.perf_counter()
            
            total_time += (end - start)
            total_images += 1
    
    return total_time / total_images if total_images > 0 else 0.0

# ========================
# VERIFICATION FUNCTION
# ========================
def verify_timing_accuracy(loader, device):
    """Verify timing accuracy by comparing multiple measurement methods."""
    print("\n" + "="*50)
    print("VERIFYING TIMING ACCURACY")
    print("="*50)
    
    # Get a single image
    noisy_img, clean_img = next(iter(loader))
    nz_np = noisy_img[0].cpu().squeeze().numpy()
    
    # Test TV Chambolle
    print("\nTV Chambolle Denoising:")
    tv_weight = 0.10
    
    # Method 1: Simple single measurement
    start = time.perf_counter()
    result1 = denoise_tv_chambolle(nz_np, weight=tv_weight)
    result1 = np.clip(result1, 0.0, 1.0)
    end = time.perf_counter()
    single_time = (end - start) * 1000
    
    # Method 2: Multiple measurements average
    times = []
    for _ in range(5):
        start = time.perf_counter()
        result2 = denoise_tv_chambolle(nz_np, weight=tv_weight)
        result2 = np.clip(result2, 0.0, 1.0)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    avg_time = np.mean(times)
    
    # Method 3: Check if denoising actually happened
    diff = np.abs(result1 - nz_np).mean()
    
    print(f"  Single measurement: {single_time:.2f} ms")
    print(f"  Average of 5 runs: {avg_time:.2f} ms (std: {np.std(times):.2f} ms)")
    print(f"  Mean pixel change: {diff:.6f}")
    
    if diff < 0.001:
        print("  WARNING: Very small change! TV might not be denoising properly.")
        print("  This explains unrealistically fast timing (2-3 ms).")
    elif avg_time < 10:
        print(f"  WARNING: {avg_time:.2f} ms is unusually fast for TV denoising!")
        print("  Expected: 50-150 ms for 128x128 image")
    elif avg_time < 50:
        print(f"  ✓ TV timing is reasonable but on the faster side ({avg_time:.2f} ms)")
    else:
        print(f"  ✓ TV timing is realistic ({avg_time:.2f} ms)")
    
    return avg_time

# ========================
# MAIN BENCHMARK RUNNER
# ========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set up paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "drp"))
    test_path = os.path.join(base_dir, "DCGAN", "test_split")
    quantum_ckpt_path = os.path.join(base_dir, "DCGAN", "models", "8-3", "all-results", "amplitude_quantum_4qubit_ckpt", "best.pth")
    classical_ckpt_path = os.path.join(base_dir, "DCGAN", "models", "8-3", "all-results", "classical_ckpt", "best.pth")
    results_json_path = os.path.join(base_dir, "DCGAN", "models", "8-3", "all-results", "evaluation_results_4qubit_vs_classical.json")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Load test dataset
    test_dataset = TestDatasetLoader(test_path, transform, 25)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    num_test_images = min(5, len(test_dataset))
    if len(test_dataset) > num_test_images:
        test_dataset = Subset(test_dataset, list(range(num_test_images)))
    
    quantum_model = None  # Ensure variable exists
    classical_model = None  # Ensure variable exists
    dncnn_model = None  # Ensure variable exists
    # Directory for saving sample images

    sample_dir = os.path.join(base_dir, "DCGAN", "models", "8-3", "all-results", "denoised_samples")
    os.makedirs(sample_dir, exist_ok=True)

    # Get a single sample (noisy, clean) for visual inspection
    sample_noisy_img, sample_clean_img = next(iter(test_loader))

    print(f"Image size: 128x128, Batch size: {test_loader.batch_size}")
    
    # Verify timing accuracy first
    tv_verify_time = verify_timing_accuracy(test_loader, device)
    
    # ========================================================
    # INFERENCE TIME BENCHMARKING
    # ========================================================
    
    results = {}
    
    # 1. Quantum 4-Qubit Model
    print("\n" + "="*50)
    print("Timing: 4-Qubit Amplitude Quantum Model")
    print("="*50)
    # Initialize quantum model; ensure variable exists even if measurement fails
    quantum_model = GeneratorQuantum4().to(device)
    q_time = measure_inference_time_model(quantum_model, quantum_ckpt_path, test_loader, device)
    if q_time:
        print(f"  [OK] Avg inference time per image: {q_time*1000:.2f} ms")
        results['quantum'] = q_time
    else:
        print("  [FAIL] Failed to measure quantum model")
    # del quantum_model  # Deletion moved later after image saving
    if device == "cuda":
        torch.cuda.empty_cache()

    # 2. Classical U-Net
    print("\n" + "="*50)
    print("Timing: Classical U-Net Model")
    print("="*50)
    if os.path.exists(classical_ckpt_path):
        classical_model = GeneratorClassical().to(device)
        c_time = measure_inference_time_model(classical_model, classical_ckpt_path, test_loader, device)
        if c_time:
            print(f"  [OK] Avg inference time per image: {c_time*1000:.2f} ms")
            results['classical_unet'] = c_time
        else:
            print("  [FAIL] Failed to measure classical model")
        # del classical_model  # Deletion moved later after image saving
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"  ✗ Classical checkpoint not found at: {classical_ckpt_path}")
        c_time = None

    # 3. DnCNN
    print("\n" + "="*50)
    print("Timing: DnCNN Model (Pretrained)")
    print("="*50)
    dncnn_model = None
    dn_time = None
    try:
        dncnn_model = get_dncnn_model(25, device)
        dn_time = measure_inference_time_dncnn(dncnn_model, test_loader, device)
        if dn_time:
            print(f"  [OK] Avg inference time per image: {dn_time*1000:.2f} ms")
            results['dncnn'] = dn_time
        else:
            print("  ✗ Failed to measure DnCNN")

        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [FAIL] Error loading DnCNN: {e}")

    # 4. BM3D
    print("\n" + "="*50)
    print("Timing: BM3D Denoising")
    print("="*50)
    bm_time = measure_inference_time_bm3d(test_loader, sigma=25, max_images=3, num_warmup=2)
    if bm_time:
        print(f"  [OK] Avg inference time per image: {bm_time*1000:.2f} ms")
        results['bm3d'] = bm_time
    else:
        print("  [FAIL] Failed to measure BM3D")

    # 5. TV Chambolle
    print("\n" + "="*50)
    print("Timing: TV Chambolle Denoising")
    print("="*50)
    tv_time = measure_inference_time_tv(test_loader, sigma=25, num_warmup=3)
    if tv_time:
        print(f"  [OK] Avg inference time per image: {tv_time*1000:.2f} ms")
        results['tv'] = tv_time
    else:
        print("  [FAIL] Failed to measure TV")

    # -----------------------------------------------------
    # Save denoised sample images for visual inspection
    # -----------------------------------------------------
    import torchvision.utils as vutils
    sample_dir = os.path.join(base_dir, "DCGAN", "models", "8-3", "all-results", "denoised_samples")
    os.makedirs(sample_dir, exist_ok=True)
    # Get a single sample (noisy, clean)
    noisy_img, clean_img = next(iter(test_loader))
    # Save ground truth and noisy side‑by‑side
    vutils.save_image(noisy_img, os.path.join(sample_dir, "noisy.png"))
    print(f"  [INFO] Saved noisy image to {os.path.join(sample_dir, 'noisy.png')}")
    vutils.save_image(clean_img, os.path.join(sample_dir, "ground_truth.png"))
    print(f"  [INFO] Saved ground truth image to {os.path.join(sample_dir, 'ground_truth.png')}")
    # Function to save a denoised result
    def _save_denosed(name, tensor):
        path = os.path.join(sample_dir, f"{name}.png")
        vutils.save_image(tensor, path)
        print(f"  [INFO] Saved {name} denoised image to {path}")
    # Quantum model
    if q_time and quantum_model is not None:
        quantum_model.eval()
        with torch.no_grad():
            den_q = quantum_model(noisy_img.to(device)).cpu()
        _save_denosed("quantum", den_q)
    # Classical U‑Net
    if c_time:
        classical_model.eval()
        with torch.no_grad():
            den_c = classical_model(noisy_img.to(device)).cpu()
        _save_denosed("classical_unet", den_c)
    # DnCNN
    if dn_time:
        dncnn_model.eval()
        with torch.no_grad():
            den_dn = dncnn_model(noisy_img.to(device)).cpu()
        _save_denosed("dncnn", den_dn)
    # BM3D
    if bm_time:
        # Use the same BM3D pipeline as in measure_inference_time_bm3d
        sample_np = noisy_img[0].cpu().squeeze().numpy()
        sigma_norm = 25 / 255.0
        den_bm = bm3d.bm3d(sample_np, sigma_psd=sigma_norm, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        den_bm = np.clip(den_bm, 0.0, 1.0)
        den_bm_tensor = torch.from_numpy(den_bm).float().unsqueeze(0).unsqueeze(0)
        _save_denosed("bm3d", den_bm_tensor)
    # TV Chambolle
    if tv_time:
        tv_weight = 0.10 if 25 == 25 else 0.05
        sample_np = noisy_img[0].cpu().squeeze().numpy()
        den_tv = denoise_tv_chambolle(sample_np, weight=tv_weight)
        den_tv = np.clip(den_tv, 0.0, 1.0)
        den_tv_tensor = torch.from_numpy(den_tv).float().unsqueeze(0).unsqueeze(0)
        _save_denosed("tv_chambolle", den_tv_tensor)

    # ========================================================
    # PRINT INFERENCE TIME SUMMARY TABLE
    # ========================================================
    print("\n" + "="*80)
    print(f"{'Model / Denoising Method':<35} | {'Time (ms)':<15} | {'Time (s)':<15} | {'Relative Speed':<15}")
    print("-"*80)
    
    # Find fastest for relative comparison
    times_ms = {k: v*1000 for k, v in results.items() if v}
    if times_ms:
        fastest = min(times_ms.values())
        fastest_name = min(times_ms, key=times_ms.get)
        
        for name, key in [
            ("4-Qubit Amplitude Quantum", 'quantum'),
            ("Classical U-Net", 'classical_unet'),
            ("DnCNN (Pretrained)", 'dncnn'),
            ("BM3D (Classical)", 'bm3d'),
            ("TV Chambolle (Classical)", 'tv'),
        ]:
            if key in results:
                t = results[key]
                t_ms = t * 1000
                rel_speed = fastest / t_ms
                print(f"{name:<35} | {t_ms:<15.2f} | {t:<15.6f} | {rel_speed:<15.2f}x")
            else:
                print(f"{name:<35} | {'FAILED':<15} | {'FAILED':<15} | {'N/A':<15}")
        
        print("-"*80)
        print(f"Fastest method: {fastest_name} ({fastest:.2f} ms per image)")
    else:
        print("No successful timing measurements!")
    
    print("="*80 + "\n")

    # ========================================================
    # MERGE INFERENCE TIMES INTO EXISTING RESULTS JSON
    # ========================================================
    if os.path.exists(results_json_path):
        with open(results_json_path, "r") as f:
            out_results = json.load(f)
    else:
        out_results = {}

    for key, t in results.items():
        if key not in out_results:
            out_results[key] = {}
        out_results[key]["avg_inference_time_seconds"] = t
        out_results[key]["avg_inference_time_ms"] = t * 1000

    with open(results_json_path, "w") as f:
        json.dump(out_results, f, indent=4)
    print(f"[OK] Inference times saved to {results_json_path}")


if __name__ == "__main__":
    main()
