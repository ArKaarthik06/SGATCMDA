import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from ..data.dataset_loader import DatasetLoader

class GraphProximityAuditor:
    """
    Computes bipartite graph topological proximity metrics (Addressing Reviewer Comment 6):
      - Shortest path distance in the known bipartite HMDAD network
      - Shared neighbor count
      - Evaluates whether predicted novel associations are trivial local neighbors or novel multi-hop connections.
    """
    def __init__(self, dataset: DatasetLoader):
        self.dataset = dataset
        self.G = nx.Graph()
        self._build_bipartite_graph()

    def _build_bipartite_graph(self):
        # Nodes: disease IDs 'D_0'...'D_38', microbe IDs 'M_0'...'M_291'
        for i in range(self.dataset.nd):
            self.G.add_node(f"D_{i}", bipartite=0, name=self.dataset.i2d[i])
        for j in range(self.dataset.nm):
            self.G.add_node(f"M_{j}", bipartite=1, name=self.dataset.i2m[j])
            
        pos_d, pos_m = self.dataset.pos_d_all, self.dataset.pos_m_all
        for d, m in zip(pos_d, pos_m):
            self.G.add_edge(f"D_{d}", f"M_{m}")

    def analyze_pair_proximity(self, disease_idx: int, microbe_idx: int) -> Dict:
        """
        Analyze distance and shared topological context for a specific (disease, microbe) pair.
        """
        d_node = f"D_{disease_idx}"
        m_node = f"M_{microbe_idx}"
        
        # Shortest path
        try:
            sp_len = nx.shortest_path_length(self.G, source=d_node, target=m_node)
        except nx.NetworkXNoPath:
            sp_len = -1  # Disconnected
            
        # Known microbes for this disease
        d_microbes = set(self.G.neighbors(d_node))
        
        # Diseases associated with this microbe
        m_diseases = set(self.G.neighbors(m_node))
        
        # Shared 2-hop connection: diseases that share microbes with disease d
        shared_disease_hops = 0
        for md in m_diseases:
            common_microbes = d_microbes.intersection(set(self.G.neighbors(md)))
            shared_disease_hops += len(common_microbes)

        return {
            'disease': self.dataset.i2d[disease_idx],
            'microbe': self.dataset.i2m[microbe_idx],
            'shortest_path_length': sp_len,
            'is_known': self.dataset.A[disease_idx, microbe_idx] == 1.0,
            'target_microbe_known_diseases': len(m_diseases),
            'target_disease_known_microbes': len(d_microbes),
            'shared_intermediate_associations': shared_disease_hops,
            'discovery_type': 'Direct' if sp_len == 1 else ('3-Hop Shared Context' if sp_len == 3 else 'Distant / Multi-hop (>3)')
        }

    def audit_top_predictions(self, predictions_df) -> List[Dict]:
        """
        Audit proximity for top novel predictions across diseases.
        """
        proximity_records = []
        for _, row in predictions_df.iterrows():
            d_name = row['Disease']
            m_name = row['Microbe']
            if d_name in self.dataset.d2i and m_name in self.dataset.m2i:
                d_idx = self.dataset.d2i[d_name]
                m_idx = self.dataset.m2i[m_name]
                info = self.analyze_pair_proximity(d_idx, m_idx)
                info['score'] = float(row.get('Score', 0.0))
                info['rank'] = int(row.get('Rank', 0))
                proximity_records.append(info)
        return proximity_records
