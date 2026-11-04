import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import KFold

class DatasetLoader:
    def __init__(self, data_path: str, seed: int = 42):
        self.data_path = data_path
        self.seed = seed
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_uniq: Optional[pd.DataFrame] = None
        self.diseases: List[str] = []
        self.microbes: List[str] = []
        self.d2i: Dict[str, int] = {}
        self.m2i: Dict[str, int] = {}
        self.i2d: Dict[int, str] = {}
        self.i2m: Dict[int, str] = {}
        self.nd: int = 0
        self.nm: int = 0
        self.A: Optional[np.ndarray] = None
        self.pos_d_all: Optional[np.ndarray] = None
        self.pos_m_all: Optional[np.ndarray] = None
        self.neg_d_all: Optional[np.ndarray] = None
        self.neg_m_all: Optional[np.ndarray] = None
        
        self._load()

    def _load(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found at {self.data_path}")
        
        self.df_raw = pd.read_csv(self.data_path, sep='\t')
        self.df_uniq = self.df_raw.drop_duplicates(subset=['Disease', 'Microbe']).reset_index(drop=True)
        
        self.diseases = sorted(self.df_uniq['Disease'].unique())
        self.microbes = sorted(self.df_uniq['Microbe'].unique())
        self.d2i = {d: i for i, d in enumerate(self.diseases)}
        self.m2i = {m: i for i, m in enumerate(self.microbes)}
        self.i2d = {i: d for i, d in enumerate(self.diseases)}
        self.i2m = {i: m for i, m in enumerate(self.microbes)}
        self.nd = len(self.diseases)
        self.nm = len(self.microbes)
        
        self.A = np.zeros((self.nd, self.nm), dtype=np.float32)
        for _, row in self.df_uniq.iterrows():
            d_idx = self.d2i[row['Disease']]
            m_idx = self.m2i[row['Microbe']]
            self.A[d_idx, m_idx] = 1.0
            
        self.pos_d_all, self.pos_m_all = np.where(self.A == 1)
        self.neg_d_all, self.neg_m_all = np.where(self.A == 0)

    def get_kfold_splits(self, k_folds: int = 5, shuffle_seed: Optional[int] = None) -> List[Dict]:
        seed = self.seed if shuffle_seed is None else shuffle_seed
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        splits = []
        pos_indices = np.arange(len(self.pos_d_all))
        
        for fold, (tr_idx, te_idx) in enumerate(kf.split(pos_indices)):
            tr_pd = self.pos_d_all[tr_idx]
            tr_pm = self.pos_m_all[tr_idx]
            te_pd = self.pos_d_all[te_idx]
            te_pm = self.pos_m_all[te_idx]
            
            # Construct fold-specific training adjacency matrix
            A_tr = np.zeros((self.nd, self.nm), dtype=np.float32)
            A_tr[tr_pd, tr_pm] = 1.0
            
            # Balanced test negative sampling
            A_te = np.zeros((self.nd, self.nm), dtype=np.float32)
            A_te[te_pd, te_pm] = 1.0
            neg_td, neg_tm = np.where((self.A == 0) & (A_te == 0))
            
            rng = np.random.RandomState(seed + fold + 100)
            nidx = rng.choice(len(neg_td), size=len(te_pd), replace=False)
            te_neg_d = neg_td[nidx]
            te_neg_m = neg_tm[nidx]
            
            splits.append({
                'fold': fold,
                'tr_pd': tr_pd,
                'tr_pm': tr_pm,
                'te_pd': te_pd,
                'te_pm': te_pm,
                'te_neg_d': te_neg_d,
                'te_neg_m': te_neg_m,
                'A_tr': A_tr
            })
        return splits

    def summary(self) -> Dict:
        return {
            'num_diseases': self.nd,
            'num_microbes': self.nm,
            'num_associations': len(self.pos_d_all),
            'sparsity_density_pct': float(self.A.mean() * 100),
            'num_negative_pairs': len(self.neg_d_all)
        }
