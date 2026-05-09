import xgboost as xgb
import numpy as np

class DummySpecialist:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def predict_proba(self, X):
        res = np.zeros((len(X), self.num_classes))
        return res

def train_specialist(X_train, y_train_subset, num_expected_species):
    y_ints_raw = np.argmax(y_train_subset, axis=1)
    present_species_indices = np.unique(y_ints_raw)
    
    label_to_local = {old: new for new, old in enumerate(present_species_indices)}
    y_local = np.array([label_to_local[val] for val in y_ints_raw])
    
    num_present = len(present_species_indices)

    model = xgb.XGBClassifier(
        objective='multi:softprob' if num_present > 2 else 'binary:logistic',
        n_estimators=400,
        max_depth=6,
        n_jobs=1,
        random_state=42
    )
    
    model.fit(X_train, y_local)
    
    model.present_species_indices = present_species_indices
    model.total_expected = num_expected_species
    
    return model