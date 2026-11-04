"""Aggregates every result file into the final tables (markdown + json). No numbers are typed by hand."""
import json, collections, numpy as np
from scipy.stats import wilcoxon
O = 'sgat_clmda_outputs/'
ARCH = [('parallel', 'SGAT-CLMDA (parallel GCN || SAGA)'), ('serial', 'GCATCMDA-style (serial GCN->GAT)'), ('gcn_only', 'GCN-only encoder'),
        ('gat_only', 'GAT-only encoder'), ('viewA_only', 'SGAT-CLMDA without View B / contrastive')]
M = {a: json.load(open(f'{O}matched_{a}.json')) for a, _ in ARCH}
def col(a, k, i=0): return np.array([v[k][i] for _, v in sorted(M[a].items(), key=lambda kv: (int(kv[0].split('|')[1]), int(kv[0].split('|')[2])))])
def ms(x): return f'{x.mean():.4f} ± {x.std():.4f}'
def wp(a, b): return wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
out, L = {}, []
L.append('## Table A. Matched protocols, identical folds / negatives / seeds (5 folds x 5 seeds = 25 runs per cell; mean ± sd)\n')
L.append('| Model | Leaky AUC | Honest AUC | gap | p (leaky vs honest) | Honest+masked AUC | Honest+masked+CF AUC | Honest+masked+CF AUPR |')
L.append('|---|---|---|---|---|---|---|---|')
for a, n in ARCH:
    lk, ho, mk, hy, hp = col(a, 'leaky'), col(a, 'honest'), col(a, 'masked'), col(a, 'hybrid'), col(a, 'hybrid', 1)
    out[a] = dict(n=len(lk), leaky=ms(lk), honest=ms(ho), masked=ms(mk), hybrid=ms(hy), hybrid_aupr=ms(hp), gap=float((lk - ho).mean()))
    L.append(f'| {n} | {ms(lk)} | {ms(ho)} | {(lk-ho).mean():+.3f} | {wp(lk, ho):.1e} | {ms(mk)} | **{ms(hy)}** | {ms(hp)} |')
cf, cfp = col('parallel', 'cf'), col('parallel', 'cf', 1)
L.append(f'| Item-based CF, no learning (A_train only) | - | {ms(cf)} | - | - | - | {ms(cf)} | {ms(cfp)} |')
cb = json.load(open(O + 'classical_baselines.json'))
for k, v in cb.items(): L.append(f"| {k} (honest, re-implemented) | - | {v['auc']:.4f} ± {v['auc_sd']:.4f} | - | - | - | - | {v['aupr']:.4f} |")
L.append('\nDefinitions: *Leaky* = GIP/cosine similarities from the full matrix. *Honest* = recomputed from A_train in every fold. *Masked* = honest + masked-edge training (mask ratio 0.5, fixed a priori). '
         '*+CF* = 0.5 x rank(model) + 0.5 x rank(item-CF), weight fixed a priori. Nothing was tuned on test folds.\n')

L.append('## Table B. Paired Wilcoxon tests on AUC (25 paired runs)\n'); L.append('| Comparison | mean diff | p | wins/25 |'); L.append('|---|---|---|---|')
def pair(name, x, y):
    L.append(f'| {name} | {np.mean(x-y):+.4f} | {wp(x, y):.1e} | {(x>y).sum()} |'); out.setdefault('paired', {})[name] = dict(diff=float(np.mean(x - y)), p=float(wp(x, y)))
H = {a: col(a, 'hybrid') for a, _ in ARCH}; K = {a: col(a, 'masked') for a, _ in ARCH}
pair('Final (parallel, masked+CF) vs item-CF alone', H['parallel'], cf)
pair('Final (parallel, masked+CF) vs masked parallel model alone', H['parallel'], K['parallel'])
kz = np.array(cb['KATZ-style']['per_run_auc']); pair('Final vs KATZ-style (honest)', H['parallel'], kz)
for a, n in ARCH[1:]:
    pair(f'masked+CF: parallel vs {n}', H['parallel'], H[a]); pair(f'masked only: parallel vs {n}', K['parallel'], K[a])
pair('masked only: parallel vs View-A-only (no View B)', K['parallel'], K['viewA_only'])
for a, n in ARCH: pair(f'{n}: honest+masked+CF vs honest plain', H[a], col(a, 'honest'))

L.append('\n## Table C. Cost (CPU, 3 threads, one fold; plain / masked training)\n'); L.append('| Model | Parameters | ms/epoch plain | ms/epoch masked | peak RSS MB (process) |'); L.append('|---|---|---|---|---|')
for a, n in ARCH:
    v = list(M[a].values())
    L.append(f"| {n} | {v[0]['params']:,} | {np.mean([x['ms_per_epoch_plain'] for x in v]):.0f} | {np.mean([x['ms_per_epoch_masked'] for x in v]):.0f} | {max(x.get('rss_mb', 0) for x in v):.0f} |")

L.append('\n## Table D. Non-learning probes (Reviewer 4 #2) and leakage audit, 25 runs\n'); L.append('| Probe | AUC | AUPR |'); L.append('|---|---|---|')
for k, v in json.load(open(O + 'leakage_multiseed.json')).items(): L.append(f"| {k} | {v['auc']:.4f} ± {v['auc_sd']:.4f} | {v['aupr']:.4f} |")

L.append('\n## Table E. Ablations (honest + masked training, 25 runs each; mean AUC)\n'); L.append('| Variant | AUC | AUPR |'); L.append('|---|---|---|')
e = json.load(open(O + 'extra_ablations.json')); c = collections.defaultdict(list)
for k, v in e.items(): c[v['config']].append((v['auc'], v['aupr']))
mb = json.load(open(O + 'masked_benchmarks.json')); c2 = collections.defaultdict(list)
for k, v in mb.items():
    if 'mask0.5' in k: c2[v['config']].append((v['auc'], v['aupr']))
ma = json.load(open(O + 'masked_ablation.json')); c3 = collections.defaultdict(list)
for k, v in ma.items(): c3[k.split('|')[0]].append((v['auc'], v['aupr']))
for d in (c2, c, c3):
    for k, v in d.items(): L.append(f'| {k} | {np.mean([x[0] for x in v]):.4f} ± {np.std([x[0] for x in v]):.4f} | {np.mean([x[1] for x in v]):.4f} |')
v = [(x['masked'][0], x['masked'][1]) for x in M['viewA_only'].values()]
L.append(f'| View A only (no View B, no CL) | {np.mean([x[0] for x in v]):.4f} ± {np.std([x[0] for x in v]):.4f} | {np.mean([x[1] for x in v]):.4f} |')
L.append('')
open('FINAL_RESULTS_TABLES.md', 'w', encoding='utf8').write('\n'.join(L)); json.dump(out, open(O + 'final_summary.json', 'w'), indent=1)
print('\n'.join(L))
