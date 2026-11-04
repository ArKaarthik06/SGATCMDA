"""Loads every result file and exposes formatted numbers (so documents never contain hand-typed results)."""
import json, glob, os, collections, numpy as np
from scipy.stats import wilcoxon
O = 'sgat_clmda_outputs/'
J = lambda f: json.load(open(O + f))
def ms(x, d=3): x = np.asarray(x, float); return f'{x.mean():.{d}f} ± {x.std():.{d}f}'
def m(x, d=3): return f'{np.mean(x):.{d}f}'
def wp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
def pf(p): return f'{p:.1e}' if p < 1e-2 else f'{p:.2f}'
def order(d, i1=1, i2=2): return [d[k] for k in sorted(d, key=lambda k: (int(k.split('|')[i1]), int(k.split('|')[i2])))]

ARCH = [('parallel', 'SGAT-CLMDA (parallel GCN || NGA)'), ('serial', 'GCATCMDA-style (serial GCN→GAT)'), ('gcn_only', 'GCN-only encoder'),
        ('gat_only', 'GAT-only encoder'), ('viewA_only', 'SGAT-CLMDA without View B / contrastive loss')]
MT = {a: order(J(f'matched_{a}.json')) for a, _ in ARCH}
def col(a, k, i=0): return np.array([r[k][i] for r in MT[a]])
CB = J('classical_baselines.json'); LM = J('leakage_multiseed.json'); LR = J('lr_probe.json')
EA, MA, MB = J('extra_ablations.json'), J('masked_ablation.json'), J('masked_benchmarks.json')
CS, KN = J('case_study_summary.json'), J('key_numbers.json'); PT = J('paired_tests.json')
REP = {n: order(json.load(open(f'{O}gcatcmda_replica_{n}.json')), 0, 1) for n in ['original', 'perfold', 'original_lr0.001', 'perfold_lr0.001']}
ROB = {}
for f in sorted(glob.glob(O + 'robustness_*.json')): ROB.update(json.load(open(f)))
ROB = order(ROB, 0, 1) if ROB else []
DIS = {}
for f in sorted(glob.glob(O + 'disbiome_[0-9]*.json')): DIS.update(json.load(open(f)))
DIS = order(DIS, 0, 1) if DIS else []
def group(d, key=lambda k, v: v['config']):
    g = collections.defaultdict(list)
    for k, v in d.items(): g[key(k, v)].append(v)
    return g
def aucs(lst): return np.array([v['auc'] for v in lst]);
