import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

def load_ppi_data(file_path):
    df = pd.read_csv(file_path, sep='\t', comment='#')
    protein_a = df.iloc[:, 0]
    protein_b = df.iloc[:, 1]
    all_proteins = pd.concat([protein_a, protein_b]).unique()
    le = LabelEncoder().fit(all_proteins)

    edge_index = torch.tensor([
        le.transform(protein_a),
        le.transform(protein_b)
    ], dtype=torch.long)

    num_nodes = len(le.classes_)
    x = torch.eye(num_nodes)

    data = Data(x=x, edge_index=edge_index)
    return data, le
