import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_metrics(json_path: Path, order):
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return [data[key] for key in order]


def draw_bar_chart(labels, values, colors, title, ylabel, output_path, y_range=None, label_format=None):
    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2, alpha=0.88)

    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        if label_format is not None:
            label_text = label_format(value)
        else:
            label_text = f"{value:.3f}" if abs(value) < 1 else f"{value:.2f}"
        plt.text(x, y + (0.01 * (y_range[1] if y_range else max(values))),
                 label_text, ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Models', fontsize=16, fontweight='bold')
    plt.ylabel(ylabel, fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, rotation=18)
    plt.yticks(fontsize=14)

    if y_range:
        plt.ylim(y_range)

    plt.grid(axis='y', linestyle='--', alpha=0.4)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    base_dir = Path(r"c:\Users\Gurung\Desktop\drp\DCGAN\models\8-3")
    json_path = base_dir / "all-results" / "evaluation_results_4qubit_vs_classical.json"
    output_dir = base_dir / "all-results"

    labels = [
        "TV",
        "BM3D",
        "DnCNN",
        "UNet based GAN",
        "Unet based Quantum enhanced GAN",
    ]
    order = ["tv", "bm3d", "dncnn", "classical_unet", "quantum"]
    colors = ['#CFD8DC', '#90A4AE', '#607D8B', '#E53935', '#1E88E5']

    metrics = load_metrics(json_path, order)
    fid_values = [m['fid'] for m in metrics]
    lpips_values = [m['lpips'] for m in metrics]

    fid_min = min(fid_values)
    fid_max = max(fid_values)
    fid_margin_low = max(0, fid_min - 5)
    fid_margin_high = fid_max + 20

    draw_bar_chart(
        labels,
        fid_values,
        colors,
        "FID Comparison Across Methods",
        "FID (lower is better)",
        output_dir / "evaluation_fid_comparison.png",
        y_range=(fid_margin_low, fid_margin_high),
        label_format=lambda v: f"{v:.1f}",
    )

    draw_bar_chart(
        labels,
        lpips_values,
        colors,
        "LPIPS Comparison Across Methods",
        "LPIPS (lower is better)",
        output_dir / "evaluation_lpips_comparison.png",
        y_range=(0.03, 0.20),
        label_format=lambda v: f"{v:.3f}",
    )


if __name__ == '__main__':
    main()
