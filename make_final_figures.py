import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O='sgat_clmda_outputs/'
L=json.load(open(O+'leakage_multiseed.json')); b=pd.DataFrame(json.load(open(O+'real_benchmarks.json')).values()); m=pd.DataFrame(json.load(open(O+'masked_benchmarks.json')).values())
cs=json.load(open(O+'case_study_summary.json'))
mm=lambda c: m[m.config==c].auc
mask_par=mm('SGAT-CLMDA parallel + masked-edge training (r=0.5)'); mask_ser=mm('Serial GCN->GAT + masked-edge training (r=0.5)')
# Fig1
items=[('Raw A lookup\n(no model)',L['heur_a_raw_A']['auc'],L['heur_a_raw_A']['auc_sd'],'#c0392b'),('Full-GIP projection\n(no model)',L['heur_b_full_GIP_projection']['auc'],L['heur_b_full_GIP_projection']['auc_sd'],'#c0392b'),
('Item-CF, full-A\nsimilarity',L['cf_leaky_(A_full x S_M_full)']['auc'],L['cf_leaky_(A_full x S_M_full)']['auc_sd'],'#c0392b'),('SGAT-CLMDA, full-A\nsimilarity (leaky)',L['model_leaky_features']['auc'],L['model_leaky_features']['auc_sd'],'#c0392b'),
('Item-CF, per-fold\nsimilarity',L['cf_honest_(A_tr x S_M_tr)']['auc'],L['cf_honest_(A_tr x S_M_tr)']['auc_sd'],'#1f77b4'),('SGAT-CLMDA, per-fold\n+ masked-edge training',mask_par.mean(),mask_par.std(),'#1f77b4'),
('SGAT-CLMDA, per-fold\n(plain training)',L['model_honest_features']['auc'],L['model_honest_features']['auc_sd'],'#1f77b4'),('Fold-GIP projection\n(no model)',L['heur_c_fold_GIP_projection']['auc'],L['heur_c_fold_GIP_projection']['auc_sd'],'#1f77b4')]
fig,ax=plt.subplots(figsize=(9,4.6)); x=np.arange(len(items)); ax.bar(x,[i[1] for i in items],yerr=[i[2] for i in items],color=[i[3] for i in items],capsize=3)
ax.axhline(.5,ls=':',c='gray'); ax.set_xticks(x); ax.set_xticklabels([i[0] for i in items],fontsize=7,rotation=0); ax.set_ylabel('AUC (mean ± SD, 5 seeds × 5 folds)'); ax.set_ylim(.3,1.05)
for xi,i in zip(x,items): ax.text(xi,i[1]+i[2]+.012,f'{i[1]:.3f}',ha='center',fontsize=8)
ax.set_title('Red: similarities computed from full A (leaky). Blue: recomputed from A_train only (honest)',fontsize=9); plt.tight_layout(); plt.savefig('figures/fig1_leakage_audit.png',dpi=200); plt.close()
# Fig2
rows=[('GCN-only','GCN-only'),('GAT-only','GAT-only'),('Serial\nGCN→GAT','Serial GCN->GAT (GCATCMDA-style)'),('Parallel\n(ANR)','SGAT-CLMDA (parallel, ANR)'),('Parallel\n+DEG','Parallel + DEG (edge gate)')]
fig,ax=plt.subplots(1,2,figsize=(10,4)); v=[b[b.config==r[1]].auc for r in rows]
ax[0].bar(range(5),[a.mean() for a in v],yerr=[a.std()/5 for a in v],capsize=3,color='#7f8c8d'); ax[0].set_xticks(range(5)); ax[0].set_xticklabels([r[0] for r in rows],fontsize=8); ax[0].set_ylim(.5,.9); ax[0].set_ylabel('AUC'); ax[0].set_title('(a) Plain training: all variants indistinguishable',fontsize=9)
ax[1].bar([0,1],[mask_ser.mean(),mask_par.mean()],yerr=[mask_ser.std()/5,mask_par.std()/5],capsize=3,color=['#7f8c8d','#1f77b4']); ax[1].set_xticks([0,1]); ax[1].set_xticklabels(['Serial\nGCN→GAT','Parallel\n(SGAT-CLMDA)']); ax[1].set_ylim(.5,.9); ax[1].set_title('(b) Masked-edge training (r = 0.5)',fontsize=9)
plt.tight_layout(); plt.savefig('figures/fig2_architecture.png',dpi=200); plt.close()
# Fig3
ds=list(cs); fig,ax=plt.subplots(figsize=(9,4)); w=.2; x=np.arange(len(ds))
for k,(key,lab,c) in enumerate([('top10_validated','SGAT-CLMDA top-10','#1f77b4'),('popularity_top10_validated','Popularity top-10','#e67e22'),('random10_validated','Random 10','#95a5a6')]): ax.bar(x+(k-1)*w,[cs[d][key]*10 for d in ds],w,label=lab,color=c)
ax.bar(x+2*w,[cs[d]['pool_base_rate']*100 for d in ds],w,label='Whole candidate pool',color='#2c3e50'); ax.set_xticks(x+w/2); ax.set_xticklabels([d.replace('Inflammatory bowel disease(IBD)','IBD') for d in ds]); ax.set_ylabel('% matching a Disbiome genus'); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig('figures/fig3_disbiome_validation.png',dpi=200); plt.close()
# numbers
print('masked par r.5',mask_par.mean(),mask_par.std(),m[m.config.str.contains('parallel') & m.config.str.contains('0.5')].aupr.mean())
for c,g in m.groupby('config'): print(c,round(g.auc.mean(),4),round(g.auc.std(),4),round(g.aupr.mean(),4),round(g.aupr.std(),4),round(g.ms_per_epoch.mean(),1),g.epochs.mean())
print(b.groupby('config').agg(sp=('sparsity','mean')).dropna())
t=pd.read_csv(O+'benchmark_table.csv'); print(t[['config','AUC','AUC_sd_folds','AUPR','noisy_AUC','params','ms_per_epoch','peak_rss_mb']].round(4).to_string())
print(pd.read_csv(O+'case_studies_disbiome.csv').groupby('Disease').head(3).to_string())
print(json.load(open(O+'experiment_summary.json'))['branch_similarity_cka'])
