import numpy as np
from model.ecoc_specialist import DummySpecialist


def hybrid_predict_proba(X_input, gater, specialists, taxonomy_map, num_classes=234):
    tax_probs = gater.predict_proba(X_input)
    global_probs = np.zeros((X_input.shape[0], num_classes))
    
    for tax_id, specialist in specialists.items():
        target_cols = taxonomy_map[tax_id]
        
        if isinstance(specialist, DummySpecialist):
            continue
            
        local_probs = specialist.predict_proba(X_input)
        
        tax_block = np.zeros((X_input.shape[0], specialist.total_expected))
        
        for local_idx, global_in_tax_idx in enumerate(specialist.present_species_indices):
            tax_block[:, global_in_tax_idx] = local_probs[:, local_idx]
            
        global_probs[:, target_cols] = tax_block * tax_probs[:, [tax_id]]
        
    return global_probs