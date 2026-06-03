import os, sys, glob

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
os.chdir(_script_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import nibabel as nib
from scipy.ndimage import zoom

from models.unet3d_standard import UNet3D
from models.vnet import VNet3D
from models.res_att_unet3d import ResAttUNet3D
from blockage_detection import CardiacDysfunctionDetector
def _load_volume(path):
    if not os.path.exists(path):
        return None
    return nib.load(path).get_fdata().astype(np.float32)

def _normalize(volume):
    lo, hi = np.percentile(volume, 1), np.percentile(volume, 99)
    volume = np.clip(volume, lo, hi)
    if hi - lo > 0:
        volume = (volume - lo) / (hi - lo)
    return volume

TARGET_SHAPE = (128, 128, 64)

def _resize(volume, order=1):
    factors = [t / c for t, c in zip(TARGET_SHAPE, volume.shape)]
    return zoom(volume, factors, order=order)

def main():
    TEST_DIR = "E:/Thesis Dataset 2/testing"

    # Discover device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    #Load best model (ResAtt-3D-U-Net)
    model = ResAttUNet3D(in_channels=1, out_channels=1, base_filters=16)
    model_path = os.path.join(_script_dir, "best_resatt-3d-u-net.pth")
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )
        print(f"[+] Loaded ResAtt-3D-U-Net from {model_path}")
    else:
        print(f"[-] Weights not found at {model_path}, using random init")
    model = model.to(device)
    model.eval()

    detector = CardiacDysfunctionDetector(threshold_thickening=0.35, min_blockage_size=5)

    #Discover NOR and MINF patients
    patient_dirs = sorted(glob.glob(os.path.join(TEST_DIR, "patient*")))
    groups = {"NOR": [], "MINF": []}
    for p_dir in patient_dirs:
        info = os.path.join(p_dir, "Info.cfg")
        if not os.path.exists(info):
            continue
        with open(info) as f:
            for line in f:
                if line.startswith("Group:"):
                    g = line.split(":")[1].strip()
                    if g in groups:
                        groups[g].append(p_dir)
                    break

    print(f"\nFound {len(groups['NOR'])} NOR patients and {len(groups['MINF'])} MINF patients")

    #Run analysis
    rows = []
    with torch.no_grad():
        for group_name, dirs in groups.items():
            for p_dir in tqdm(dirs, desc=f"Analyzing {group_name}"):
                pat_id = os.path.basename(p_dir)
                # read ED/ES frame numbers
                ed_frame, es_frame = 1, 12
                info = os.path.join(p_dir, "Info.cfg")
                with open(info) as f:
                    for line in f:
                        if line.startswith("ED:"):
                            ed_frame = int(line.split(":")[1].strip())
                        elif line.startswith("ES:"):
                            es_frame = int(line.split(":")[1].strip())

                ed_path = os.path.join(p_dir, f"{pat_id}_frame{ed_frame:02d}.nii.gz")
                es_path = os.path.join(p_dir, f"{pat_id}_frame{es_frame:02d}.nii.gz")

                ed_img = _load_volume(ed_path)
                es_img = _load_volume(es_path)
                if ed_img is None or es_img is None:
                    print(f"  [!] Skipping {pat_id} (missing data)")
                    continue

                ed_img = _resize(_normalize(ed_img))
                es_img = _resize(_normalize(es_img))

                ed_t = torch.from_numpy(ed_img).unsqueeze(0).unsqueeze(0).to(device)
                es_t = torch.from_numpy(es_img).unsqueeze(0).unsqueeze(0).to(device)

                threshold = 0.4  # ResAtt threshold
                ed_pred = (torch.sigmoid(model(ed_t)) > threshold).float()[0, 0].cpu().numpy()
                es_pred = (torch.sigmoid(model(es_t)) > threshold).float()[0, 0].cpu().numpy()

                info_dict = detector.detect_blockages(ed_pred, es_pred, ed_img, es_img)

                mean_thickening = float(np.mean(info_dict['regional_thickening']))

                rows.append({
                    'patient_id': pat_id,
                    'group': group_name,
                    'mean_wall_thickening': mean_thickening * 100,
                    'ef_pct': info_dict['ef_pct'],
                    'suspicion_score': info_dict['suspicion_score'],
                    'risk_level': info_dict['risk_level'],
                    'impaired_regions': info_dict['impaired_regions'],
                    'dysfunction_rate': info_dict['dysfunction_rate'],
                    'mean_wall_thickness_ed': info_dict['mean_wall_thickness'],
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(_script_dir, "nor_vs_minf_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved results to {csv_path}")

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("NOR vs MINF COMPARATIVE SUMMARY")
    print("=" * 70)
    for g in ["NOR", "MINF"]:
        gdf = df[df['group'] == g]
        print(f"\n  {g} ({len(gdf)} patients):")
        print(f"    Mean Wall Thickening:  {gdf['mean_wall_thickening'].mean():.1f}%  "
              f"(std {gdf['mean_wall_thickening'].std():.1f}%)")
        print(f"    Mean EF:               {gdf['ef_pct'].mean():.1f}%  "
              f"(std {gdf['ef_pct'].std():.1f}%)")
        print(f"    Mean Suspicion Score:   {gdf['suspicion_score'].mean():.2f}  "
              f"(std {gdf['suspicion_score'].std():.2f})")
        print(f"    Mean Impaired Regions:  {gdf['impaired_regions'].mean():.1f}/6")
        flagged = (gdf['suspicion_score'] >= 2).sum()
        print(f"    Patients flagged (>=2):  {flagged}/{len(gdf)}")

    #Generate comparison plots
    _plot_comparison(df)

    print("\n[+] Validation complete!")


def _plot_comparison(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("NOR vs MINF — Multi-Evidence Cardiac Dysfunction Comparison",
                 fontsize=16, fontweight='bold', y=0.98)

    group_colors = {'NOR': '#2ecc71', 'MINF': '#e74c3c'}

    #Wall Thickening box plot
    ax = axes[0, 0]
    data_nor = df[df['group'] == 'NOR']['mean_wall_thickening']
    data_minf = df[df['group'] == 'MINF']['mean_wall_thickening']
    bp = ax.boxplot([data_nor, data_minf], labels=['NOR', 'MINF'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(group_colors['NOR'])
    bp['boxes'][1].set_facecolor(group_colors['MINF'])
    ax.set_ylabel('Mean Wall Thickening (%)')
    ax.set_title('Wall Thickening (AHA-averaged)', fontweight='bold')
    ax.axhline(y=35, color='red', linestyle='--', alpha=0.6, label='35% threshold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    #EF box plot
    ax = axes[0, 1]
    data_nor = df[df['group'] == 'NOR']['ef_pct']
    data_minf = df[df['group'] == 'MINF']['ef_pct']
    bp = ax.boxplot([data_nor, data_minf], labels=['NOR', 'MINF'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(group_colors['NOR'])
    bp['boxes'][1].set_facecolor(group_colors['MINF'])
    ax.set_ylabel('Ejection Fraction (%)')
    ax.set_title('Ejection Fraction', fontweight='bold')
    ax.axhline(y=45, color='red', linestyle='--', alpha=0.6, label='45% threshold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Suspicion Score box plot
    ax = axes[0, 2]
    data_nor = df[df['group'] == 'NOR']['suspicion_score']
    data_minf = df[df['group'] == 'MINF']['suspicion_score']
    bp = ax.boxplot([data_nor, data_minf], labels=['NOR', 'MINF'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(group_colors['NOR'])
    bp['boxes'][1].set_facecolor(group_colors['MINF'])
    ax.set_ylabel('Suspicion Score')
    ax.set_title('Hybrid Suspicion Score', fontweight='bold')
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.6, label='Flag threshold (≥2)')
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    #Impaired Regions bar chart
    ax = axes[1, 0]
    data_nor = df[df['group'] == 'NOR']['impaired_regions']
    data_minf = df[df['group'] == 'MINF']['impaired_regions']
    x = np.arange(7)  # 0-6 possible
    nor_counts = np.array([(data_nor == i).sum() for i in x])
    minf_counts = np.array([(data_minf == i).sum() for i in x])
    w = 0.35
    ax.bar(x - w/2, nor_counts, w, label='NOR', color=group_colors['NOR'],
           edgecolor='black')
    ax.bar(x + w/2, minf_counts, w, label='MINF', color=group_colors['MINF'],
           edgecolor='black')
    ax.set_xlabel('Impaired AHA Regions')
    ax.set_ylabel('Patient Count')
    ax.set_title('Distribution of Impaired Regions', fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    #Suspicion score distribution bar chart
    ax = axes[1, 1]
    scores = [0, 1, 2, 3, 4]
    nor_sc = [(df[(df['group'] == 'NOR') & (df['suspicion_score'] == s)].shape[0])
              for s in scores]
    minf_sc = [(df[(df['group'] == 'MINF') & (df['suspicion_score'] == s)].shape[0])
               for s in scores]
    x = np.arange(len(scores))
    ax.bar(x - w/2, nor_sc, w, label='NOR', color=group_colors['NOR'],
           edgecolor='black')
    ax.bar(x + w/2, minf_sc, w, label='MINF', color=group_colors['MINF'],
           edgecolor='black')
    ax.set_xlabel('Suspicion Score')
    ax.set_ylabel('Patient Count')
    ax.set_title('Suspicion Score Distribution', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['0\n(Normal)', '1\n(Normal)', '2\n(Suspicious)',
                        '3\n(High)', '4\n(High)'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    #Per-patient scatter: thickening vs EF
    ax = axes[1, 2]
    for g in ['NOR', 'MINF']:
        gdf = df[df['group'] == g]
        ax.scatter(gdf['mean_wall_thickening'], gdf['ef_pct'],
                  c=group_colors[g], label=g, s=80, edgecolors='black',
                  alpha=0.8, zorder=3)
    ax.axvline(x=35, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=45, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean Wall Thickening (%)')
    ax.set_ylabel('Ejection Fraction (%)')
    ax.set_title('Thickening vs EF (per patient)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(_script_dir, "nor_vs_minf_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
