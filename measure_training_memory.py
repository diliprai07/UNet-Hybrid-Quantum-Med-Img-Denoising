import os, sys, torch, psutil
import importlib.util
from torch.profiler import profile, record_function, ProfilerActivity

# ---------------------------
# Load the quantum + GAN code
# ---------------------------
model_path = r"C:\\Users\\Gurung\\Desktop\\UNet-Hybrid-Quantum-Med-Img-Denoising\\hybrid-quantum-models\\amplitude-embedding\\4-qubit\\4qubit-3layer-amplitude-qauntum.py"
spec = importlib.util.spec_from_file_location('qauntum_mod', model_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

Generator = getattr(mod, 'Generator')            # UNet generator with quantum bottleneck
Discriminator = getattr(mod, 'Discriminator')
DatasetLoader = getattr(mod, 'DatasetLoader')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Reset CUDA peak stats before profiling (no-op on CPU)
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)

# Optional: enable deterministic cuDNN behavior for reproducibility
torch.backends.cudnn.benchmark = True

# ---------------------------
# Instantiate models & optimizers
# ---------------------------
G = Generator().to(device)
D = Discriminator().to(device)
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Helper
bytes_to_mib = lambda b: b / 1024**2

# ------- 1. Model parameters (G + D) -------
param_bytes = sum(p.numel() for p in G.parameters()) * 4
param_bytes += sum(p.numel() for p in D.parameters()) * 4
param_mib = bytes_to_mib(param_bytes)
print('Param (G+D) MiB:', param_mib)

# ------- 2. Adam states (moment & variance) -------
adam_state_bytes = 2 * param_bytes  # two buffers per param (exp_avg, exp_avg_sq)
adam_state_mib = bytes_to_mib(adam_state_bytes)
print('Adam states MiB (G+D):', adam_state_mib)

# ------- 3. Quantum state‑vector overhead (4 qubits) -------
q_state_bytes = (2 ** 4) * 8  # complex64 = 8 B per amplitude
q_state_mib = bytes_to_mib(q_state_bytes)
print('Quantum state-vector MiB:', q_state_mib)

# ------- 4. Input tensors (noisy + clean) -------
batch_size = 8
noisy = torch.randn(batch_size, 1, 128, 128, device=device)
clean = torch.randn(batch_size, 1, 128, 128, device=device)
input_bytes = (noisy.numel() + clean.numel()) * 4
input_mib = bytes_to_mib(input_bytes)
print('Inputs (noisy+clean) MiB:', input_mib)

# ------- 5. Peak activation + gradient memory during one training step -------
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == 'cuda' else [ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
    # forward G
    fake = G(noisy)
    # D forward on real & fake
    pred_real = D(clean)
    pred_fake = D(fake.detach())
    # G backward
    loss_G = -pred_fake.mean()
    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()
    # D backward
    # D backward omitted for memory profiling (not needed for peak memory measurement)
    # loss_D = -(pred_real.mean() - pred_fake.mean())
    # opt_D.zero_grad()
    # loss_D.backward()
    # opt_D.step()

# Peak memory observed (includes activations + gradients). Subtract parameter memory which we already counted.
try:
    peak_bytes = max(e.self_cpu_memory_usage for e in prof.function_events)
except AttributeError:
    peak_bytes = 0

peak_mib = bytes_to_mib(peak_bytes)
print('Peak activation+gradient MiB (batch 8):', peak_mib)

# GPU peak memory (if using CUDA)
if device.type == 'cuda':
    peak_gpu_bytes = torch.cuda.max_memory_allocated(device)
    peak_gpu_mib = bytes_to_mib(peak_gpu_bytes)
    print('Peak GPU activation+gradient MiB (batch 8):', peak_gpu_mib)
else:
    peak_gpu_mib = 0.0

# ------- 6. Total estimated RAM for a training step -------
# Total estimated RAM per training step (including GPU peak if applicable)
if device.type == 'cuda':
    total_mib = param_mib + adam_state_mib + q_state_mib + input_mib + peak_gpu_mib
else:
    total_mib = param_mib + adam_state_mib + q_state_mib + input_mib + peak_mib
print('---')
print('Total estimated RAM per training step (batch 8) MiB:', total_mib)
