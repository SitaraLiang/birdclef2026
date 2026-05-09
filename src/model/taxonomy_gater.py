import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

def train_taxonomy_gater(X_train, y_tax_train, X_val, y_tax_val):
    """
    Trains a multi-class XGBoost gater with class balancing.
    """
    weights = compute_sample_weight(class_weight='balanced', y=y_tax_train)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 5,
        'tree_method': 'hist',
        'learning_rate': 0.05,
        'max_depth': 6,
        'n_estimators': 500,
        'n_jobs': 1,
        'random_state': 42
    }
    
    gater = xgb.XGBClassifier(**params)
    
    print("Gater: Training with 'balanced' sample weights...")
    gater.fit(
        X_train, y_tax_train, 
        sample_weight=weights,
        eval_set=[(X_val, y_tax_val)], 
        verbose=False
    )
    
    return gater