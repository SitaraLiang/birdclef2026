import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import xgboost as xgb

def load_data(data_dir='data/processed/final_features.csv', test_size=0.15, val_size=0.15, random_state=42):
    print(f"Loading data from {data_dir}...")
    df = pd.read_csv(data_dir)

    # 1. Identify all UNIQUE individual species IDs
    # We split every row by space and collect all unique IDs found anywhere
    all_labels_series = df['final_labels'].astype(str).str.split(' ')
    species_list = sorted(list(set([item for sublist in all_labels_series for item in sublist])))
    label_to_idx = {label: i for i, label in enumerate(species_list)}
    
    # 2. Build the Multi-Label Binary Matrix (N, 234)
    n_samples = len(df)
    n_species = len(species_list)
    y = np.zeros((n_samples, n_species))
    
    for i, labels in enumerate(all_labels_series):
        for lbl in labels:
            if lbl in label_to_idx:
                y[i, label_to_idx[lbl]] = 1

    # 3. Features (Exclude filename and the label string)
    X_cols = [col for col in df.columns if col not in ['filename', 'final_labels']]
    X = df[X_cols].values

    # 4. Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=rel_val, random_state=random_state)

    print(f"Detected {n_species} unique species.")
    return X_train, y_train, X_val, y_val, X_test, y_test, species_list


def extract_taxonomy_labels(y_matrix, species_list, mapping_file='data/taxonomy.csv'):
    mapping_df = pd.read_csv(mapping_file)
    label_to_tax_name = dict(zip(mapping_df['primary_label'].astype(str), mapping_df['class_name']))
    
    tax_names = ['Insecta', 'Reptilia', 'Amphibia', 'Mammalia', 'Aves']
    tax_to_int = {name: i for i, name in enumerate(tax_names)}
    
    # For the Gater, we just need one label per row. 
    # argmax picks the first '1' it encounters in our multi-label matrix.
    primary_indices = np.argmax(y_matrix, axis=1)
    
    y_tax_ints = []
    for idx in primary_indices:
        species_id = species_list[idx]
        tax_name = label_to_tax_name.get(species_id, 'Aves')
        y_tax_ints.append(tax_to_int.get(tax_name, 4))
        
    return np.array(y_tax_ints).astype(int)


def create_taxonomy_map(species_list, mapping_file='data/taxonomy.csv'):
    """
    Creates a dict: {tax_id_int: [list_of_global_species_indices]}
    """
    mapping_df = pd.read_csv(mapping_file)
    mapping_df['primary_label'] = mapping_df['primary_label'].astype(str)
    
    tax_names = ['Insecta', 'Reptilia', 'Amphibia', 'Mammalia', 'Aves']
    name_to_id = {name: i for i, name in enumerate(tax_names)}
    
    id_to_tax_int = dict(zip(
        mapping_df['primary_label'], 
        mapping_df['class_name'].map(name_to_id)
    ))
    
    tax_map = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    for idx, species_id in enumerate(species_list):
        tax_id = id_to_tax_int.get(species_id, 4)
        tax_map[tax_id].append(idx)
        
    return tax_map



class TQDMCallback(xgb.callback.TrainingCallback):
    def __init__(self, rounds, desc="Training"):
        self.pbar = tqdm(total=rounds, desc=desc, leave=False)

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model