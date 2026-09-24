# UNet‑Hybrid‑Quantum‑Med‑Img‑Denoising

## Project Overview
This repository implements a hybrid classical‑quantum image‑denoising pipeline based on the **Dilip & Rai et al.** paper.  It provides:
- Classical UNet baseline (`classical‑model/`)
- Quantum‑enhanced generators (amplitude and angle embeddings) under `hybrid‑quantum‑models/`
- Scripts for training, evaluation (FID/PSNR/SSIM), memory profiling, and result visualisation.

The code is organised to be **portable** – all paths are relative to the repository root, enabling reproducibility on any machine.

---

## Directory Structure
```
UNet-Hybrid-Quantum-Med-Img-Denoising/
│
├─ Dataset/                     # Training/validation/test splits
│   ├─ train_split/
│   ├─ val_split/
│   └─ test_split/
│
├─ classical-model/             # Pure classical UNet implementation
│   └─ pure-classical.py
│
├─ hybrid-quantum-models/       # Quantum‑enhanced models
│   ├─ amplitude-embedding/      # Amplitude‑based quantum encodings
│   │   ├─ 4-qubit/                # Example 4‑qubit architecture
│   │   │   └─ 4qubit-3layer-amplitude-qauntum.py
│   │   ├─ 8-qubit/                # 8‑qubit version
│   │   │   └─ 8-amplitude-quantum.py
│   │   └─ … (other qubit counts)
│   └─ angle-embedding/          # Angle‑based quantum encodings
│       └─ 12quibit/                # Example 12‑qubit architecture
│           └─ 12-quantum-classical-train.py
│
├─ evaluate_fid_psnr_ssim.py    # Evaluation of FID / PSNR / SSIM for all models
├─ measure_training_memory.py   # Memory profiling script
├─ plot_fid_lpips_comparison.py # Visual comparison of metrics
├─ requirements.txt             # Python dependencies
└─ venv/                        # Optional local virtual environment
```

---

## Reproducible Setup Guide
### 1. Prerequisites
- **Operating System**: Windows (tested on Windows 10/11) – Linux/macOS work as well.
- **Python**: 3.10 or newer.
- **Git** (optional, for cloning the repo).
- **CUDA** (optional, for GPU acceleration). The code falls back to CPU automatically.

### 2. Clone the Repository
```bash
git clone https://github.com/your‑username/UNet-Hybrid-Quantum-Med-Img-Denoising.git
cd UNet-Hybrid-Quantum-Med-Img-Denoising
```
*(If you already have the folder on your desktop, skip this step.)*

### 3. Create & Activate a Virtual Environment
```bash
# Create the env inside the repo (recommended)
python -m venv venv
# Activate (PowerShell)
.\venv\Scripts\Activate.ps1
# Activate (CMD)
venv\Scripts\activate.bat
```
The repository already contains a `venv/` folder you can reuse, but recreating ensures a clean state.

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```
Key packages include:
- `torch` (with optional CUDA) 
- `torchvision`
- `pytorch‑quantum` (or the quantum SDK used by the paper)
- `scikit‑image`, `numpy`, `matplotlib`, `bm3d`
- `tqdm`, `json`, `os`, `pathlib`

### 5. Verify the Dataset Layout
The folder `Dataset/` must contain three sub‑folders:
```
Dataset/
├─ train_split/   # training images (noisy & clean pairs)
├─ val_split/     # validation images
└─ test_split/    # test images used for evaluation
```
If you obtained the data from the authors, place it exactly as above.  The scripts locate the dataset using **relative paths**, e.g.:
```python
base_dir = Path(__file__).parent   # repository root
test_path = base_dir / "Dataset" / "test_split"
```
No hard‑coded absolute paths are needed.

### 6. Training a Model (example – 4‑qubit amplitude)
```bash
python hybrid-quantum-models/amplitude-embedding/4-qubit/4qubit-3layer-amplitude-qauntum.py
```
The script automatically reads the training split, creates the model, and saves checkpoints under:
```
Dataset/…/models/8-3/all-results/<model_name>_ckpt/best.pth
```
You can change the number of qubits by navigating to the appropriate sub‑folder.

### 7. Running Evaluation
```bash
python evaluate_fid_psnr_ssim.py
```
The script loads the test split, restores the checkpoints from the paths created in step 6, computes FID/PSNR/SSIM and writes a JSON report to:
```
.../all-results/evaluation_results_4qubit_vs_classical.json
```
All generated plots are saved under the same `all-results/` folder.

### 8. Memory Profiling (optional)
```bash
python measure_training_memory.py
```
Shows RAM/GPU consumption for a single training step and helps you size your hardware.

### 9. Visualising Metric Comparisons
```bash
python plot_fid_lpips_comparison.py
```
Creates bar charts (`evaluation_fid_comparison.png`, `evaluation_lpips_comparison.png`) using the JSON report from step 7.

---

## Tips for Full Reproducibility
1. **Record the Python environment**:
   ```bash
   pip freeze > env.txt
   ```
   Share `env.txt` alongside the code.
2. **Seed everything** – the scripts already set `torch.manual_seed(0)` and `np.random.seed(0)`.
3. **GPU vs CPU** – the code detects CUDA; for exact replication, run on the same hardware or force CPU by setting `device="cpu"`.
4. **Version control** – keep a Git tag for each paper release (e.g., `v1.0‑Dilip‑Rai`).
5. **Data checksum** – provide SHA‑256 hashes of the three dataset splits to guarantee the same images are used.

---

## References
- Dilip, S., Rai, A., *et al.* “Hybrid Quantum‑Classical UNet for Medical Image Denoising”, 2026.
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Pytorch‑Quantum: https://github.com/pytorch-quantum/pytorch-quantum
- PennyLane: https://pennylane.ai
- PennyLane Lightning: https://pennylane.ai/plugins/lightning


*End of document.*
