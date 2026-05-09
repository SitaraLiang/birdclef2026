from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, label_ranking_average_precision_score
import pandas as pd


def get_optimal_thresholds(y_true, y_probs):
    best_thresholds = []
    for i in range(y_true.shape[1]):
        y_true_bird = y_true[:, i]
        y_probs_bird = y_probs[:, i]
        
        best_f1 = -1
        best_th = 0.3
        
        for th in np.arange(0.05, 0.65, 0.05):
            current_f1 = f1_score(y_true_bird, (y_probs_bird > th).astype(int), zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_th = th
        
        best_thresholds.append(best_th)
    return np.array(best_thresholds)


def evaluate_optimized_on_probs(y_probs, y_true, thresholds=None, set_name="Set"):
    """
    Revised version of your function to handle pre-calculated probabilities.
    """
    if thresholds is None:
        preds_bin = (y_probs > 0.3).astype(int)
        type_str = "Static (0.3)"
    else:
        preds_bin = (y_probs > thresholds).astype(int)
        type_str = "Optimized"
    
    f1 = f1_score(y_true, preds_bin, average='macro', zero_division=0)
    lrap = label_ranking_average_precision_score(y_true, y_probs)
    
    print(f"\nResults on {set_name} ({type_str}):")
    print(f"Macro F1 : {f1:.4f}")
    print(f"LRAP     : {lrap:.4f}")
    return f1, lrap


