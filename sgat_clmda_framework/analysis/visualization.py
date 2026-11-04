import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.linewidth': 0.8
})
COLORS = ['#2E86AB', '#E84855', '#3BB273', '#F7B731', '#9B59B6', '#1ABC9C', '#E67E22', '#95A5A6']

class Visualizer:
    """
    Generates high-resolution, publication-ready figures for all experiments.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_roc(self, all_fpr: List, all_tpr: List, all_auc: List, save_name: str = 'plot_roc.png'):
        fig, ax = plt.subplots(figsize=(6.5, 6))
        mean_fpr = np.linspace(0, 1, 200)
        tprs = [np.interp(mean_fpr, f, t) for f, t in zip(all_fpr, all_tpr)]
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)

        for i, (fpr, tpr, a) in enumerate(zip(all_fpr, all_tpr, all_auc)):
            ax.plot(fpr, tpr, lw=1.0, alpha=0.35, color=COLORS[i % len(COLORS)], label=f'Fold {i+1} (AUC={a:.4f})')
        ax.plot(mean_fpr, mean_tpr, lw=2.5, color='black', label=f'Mean (AUC={np.mean(all_auc):.4f}±{np.std(all_auc):.4f})')
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.12, color='black', label='±1 Std Dev')
        ax.plot([0, 1], [0, 1], '--', color='grey', lw=1.0)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('SGAT-CLMDA: ROC Curves (5-Fold CV)', fontweight='bold')
        ax.legend(fontsize=8.5, loc='lower right')
        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path

    def plot_pr(self, all_prec: List, all_rec: List, all_aupr: List, save_name: str = 'plot_pr.png'):
        fig, ax = plt.subplots(figsize=(6.5, 6))
        for i, (prec, rec, a) in enumerate(zip(all_prec, all_rec, all_aupr)):
            ax.plot(rec, prec, lw=1.2, alpha=0.5, color=COLORS[i % len(COLORS)], label=f'Fold {i+1} (AUPR={a:.4f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'SGAT-CLMDA: Precision-Recall Curves\nMean AUPR = {np.mean(all_aupr):.4f} ± {np.std(all_aupr):.4f}', fontweight='bold')
        ax.legend(fontsize=8.5, loc='lower left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path

    def plot_leakage_audit(self, non_learn_res: Dict, full_vs_recalc_res: Dict, save_name: str = 'plot_leakage_audit.png'):
        """Addressing Reviewer Comments 1 and 2."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Non-learning heuristic baseline audit (Reviewer Comment 2)
        ax = axes[0]
        methods = list(non_learn_res.keys())
        auc_means = [non_learn_res[m]['auc_mean'] for m in methods]
        bars = ax.bar(methods, auc_means, color=[COLORS[1], COLORS[3], COLORS[0]], edgecolor='white', width=0.5)
        
        min_auc = min(auc_means)
        max_auc = max(auc_means)
        spread = max(max_auc - min_auc, 0.1)
        ax.set_ylim(max(0.0, np.floor((min_auc - 0.25 * spread) * 10) / 10), min(1.15, max_auc + 0.12))
        
        for bar, v in zip(bars, auc_means):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', fontsize=9.5)
        ax.set_ylabel('AUC')
        ax.set_title('(A) Non-Learning Heuristics (Reviewer Comment 2)\nLabel Lookup vs. GIP Features', fontweight='bold', fontsize=10.5)

        # Panel 2: Full vs. Recalculated Features on SGAT-CLMDA (Reviewer Comment 1)
        ax2 = axes[1]
        configs = ['Full Features (Leaky)', 'Recalculated (Honest)']
        vals = [full_vs_recalc_res['Full Similarity Features (Leaky)']['auc_mean'],
                full_vs_recalc_res['Recalculated Fold Features (Honest)']['auc_mean']]
        bars2 = ax2.bar(configs, vals, color=[COLORS[1], COLORS[2]], edgecolor='white', width=0.45)
        
        min_v = min(vals)
        max_v = max(vals)
        spread2 = max(max_v - min_v, 0.03)
        ymin2 = max(0.0, np.floor((min_v - 0.4 * spread2) * 50) / 50)
        ymax2 = min(1.05, np.ceil((max_v + 0.45 * spread2) * 50) / 50)
        ax2.set_ylim(ymin2, ymax2)
        
        for bar, v in zip(bars2, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, v + 0.015 * (ymax2 - ymin2), f'{v:.4f}', ha='center', fontweight='bold', fontsize=11)
        gap = full_vs_recalc_res.get('leakage_gap_auc', vals[0] - vals[1])
        ax2.annotate('', xy=(1, vals[1]), xytext=(0, vals[0]), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax2.text(0.5, (vals[0] + vals[1])/2 + 0.03 * (ymax2 - ymin2), f'Feature Gap = +{gap:.4f}', ha='center', fontweight='bold', fontsize=10)
        ax2.set_ylabel('AUC')
        ax2.set_title('(B) Feature Leakage Ablation (Reviewer Comment 1)\nFull A vs Per-Fold Recalculated', fontweight='bold', fontsize=10.5)

        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path

    def plot_cka_and_branches(self, cka_res: Dict, serial_vs_parallel_res: Dict, save_name: str = 'plot_cka_and_branches.png'):
        """Addressing Reviewer Comment 4."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: CKA and Cosine Similarity across layers
        ax = axes[0]
        layers = list(cka_res.keys())
        cka_vals = [cka_res[l]['linear_cka'] for l in layers]
        cos_vals = [cka_res[l]['mean_cosine'] for l in layers]
        x = np.arange(len(layers))
        w = 0.35
        b1 = ax.bar(x - w/2, cka_vals, w, label='Linear CKA', color=COLORS[0], edgecolor='white')
        b2 = ax.bar(x + w/2, cos_vals, w, label='Mean Cosine Similarity', color=COLORS[4], edgecolor='white')
        for bar, v in zip(b1, cka_vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9.5, fontweight='bold')
        for bar, v in zip(b2, cos_vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9.5, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.set_ylim(0, 1.25)
        ax.axhline(0.85, color='grey', linestyle=':', label='High Redundancy Threshold (0.85)')
        ax.set_ylabel('Similarity Metric')
        ax.set_title('(A) Branch Representation Similarity (CKA & Cosine)\nGCN vs. SAGA Branches', fontweight='bold', fontsize=10.5)
        ax.legend(fontsize=9, loc='upper left')

        # Panel 2: Serial vs. Parallel across seeds
        ax2 = axes[1]
        archs = list(serial_vs_parallel_res.keys())[:2]
        means = [serial_vs_parallel_res[a]['auc_mean'] for a in archs]
        stds = [serial_vs_parallel_res[a].get('auc_std', 0.0) for a in archs]
        bars_arch = ax2.bar(archs, means, yerr=stds, capsize=6, color=[COLORS[2], COLORS[6]], edgecolor='white', width=0.45)
        
        min_val = min([m - s for m, s in zip(means, stds)])
        max_val = max([m + s for m, s in zip(means, stds)])
        spread = max(max_val - min_val, 0.03)
        ymin = max(0.0, np.floor((min_val - 0.4 * spread) * 50) / 50)
        ymax = min(1.02, np.ceil((max_val + 0.5 * spread) * 50) / 50)
        ax2.set_ylim(ymin, ymax)
        
        for bar, v, s in zip(bars_arch, means, stds):
            ax2.text(bar.get_x() + bar.get_width()/2, v + s + 0.015 * (ymax - ymin),
                     f'{v:.4f}', ha='center', fontweight='bold', fontsize=10.5)
        ax2.set_ylabel('AUC')
        delta = serial_vs_parallel_res.get('delta_auc', means[0] - means[1] if len(means) > 1 else 0.0)
        ax2.set_title(f'(B) Parallel vs. Serial Encoder (Multiple Random Seeds)\nDelta AUC = {delta:+.4f}', fontweight='bold', fontsize=10.5)

        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path

    def plot_cl_ablation_and_temp(self, cl_res: Dict, temp_res: Dict, save_name: str = 'plot_cl_ablation_and_temp.png'):
        """Addressing Reviewer Comment 5."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Independent Disease vs Microbe CL
        ax = axes[0]
        modes_map = {
            'both': 'Full CL\n(Disease+Microbe)',
            'disease_only': 'Disease CL\nOnly',
            'microbe_only': 'Microbe CL\nOnly',
            'none': 'No CL\n(BCE Only)'
        }
        labels = [modes_map.get(k, k) for k in cl_res.keys()]
        vals = [cl_res[k]['auc_mean'] for k in cl_res.keys()]
        bars = ax.bar(labels, vals, color=[COLORS[2], COLORS[0], COLORS[3], COLORS[7]], edgecolor='white', width=0.5)
        
        min_v = min(vals)
        max_v = max(vals)
        spread = max(max_v - min_v, 0.02)
        ymin = max(0.0, np.floor((min_v - 0.4 * spread) * 50) / 50)
        ymax = min(1.02, np.ceil((max_v + 0.45 * spread) * 50) / 50)
        ax.set_ylim(ymin, ymax)
        
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.015 * (ymax - ymin),
                    f'{v:.4f}', ha='center', fontweight='bold', fontsize=10)
        ax.set_ylabel('AUC')
        ax.set_title('(A) Contrastive Learning Component Contribution\nIndependent Disease vs. Microbe Ablation', fontweight='bold', fontsize=10.5)

        # Panel 2: Temperature Scan
        ax2 = axes[1]
        temps = [temp_res[k]['tau'] for k in temp_res.keys()]
        t_aucs = [temp_res[k]['auc'] for k in temp_res.keys()]
        
        min_tau_auc = min(t_aucs)
        max_tau_auc = max(t_aucs)
        tau_spread = max(max_tau_auc - min_tau_auc, 0.015)
        ax2.set_ylim(min_tau_auc - 0.3 * tau_spread, max_tau_auc + 0.45 * tau_spread)
        
        ax2.plot(temps, t_aucs, marker='o', lw=2.2, color=COLORS[0], markersize=8)
        for t, val in zip(temps, t_aucs):
            ax2.text(t, val + 0.08 * tau_spread, f'{val:.4f}', ha='center', fontsize=9.5, fontweight='bold')
        ax2.set_xlabel('Contrastive Temperature tau_CL')
        ax2.set_ylabel('AUC')
        ax2.set_title('(B) Contrastive Temperature Hyperparameter Scan', fontweight='bold', fontsize=10.5)

        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path

    def plot_case_studies(self, case_dict: Dict[str, pd.DataFrame], save_name: str = 'plot_case_studies.png'):
        """Addressing Reviewer Comment 6 with validation source badges."""
        n_cases = len(case_dict)
        fig, axes = plt.subplots(1, n_cases, figsize=(5.2 * n_cases, 5.8))
        if n_cases == 1:
            axes = [axes]

        for ax, (disease, df) in zip(axes, case_dict.items()):
            df10 = df.head(10).copy()
            colors = [COLORS[2] if row['External Validation'] == 'Yes' else COLORS[0] for _, row in df10.iterrows()]
            y_pos = np.arange(len(df10))
            ax.barh(y_pos, df10['Mean Score'][::-1], color=colors[::-1], edgecolor='white', height=0.6)
            ax.set_yticks(y_pos)
            labels = [f"{row['Microbe'][:20]} (Top {row['Pool Percentile (%)']}%)" for _, row in df10[::-1].iterrows()]
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Prediction Score (Mean)', fontsize=9)
            short_title = disease[:24] + '...' if len(disease) > 24 else disease
            ax.set_title(short_title, fontsize=10, fontweight='bold')
            ax.set_xlim(0, 1.05)

        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=COLORS[2], label='Confirmed in Disbiome / Literature'),
                            Patch(color=COLORS[0], label='Novel Unindexed Candidate')],
                   loc='lower center', ncol=2, fontsize=10.5, bbox_to_anchor=(0.5, -0.05))
        plt.suptitle('Top-10 Novel Microbe Predictions with Blind Validation Confirmation', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path
