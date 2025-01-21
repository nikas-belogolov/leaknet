import os
import pandas as pd
from config import DATA_RAW_DIR, DATA_AUGMENTED_DIR, ROOT_DIR
import numpy as np
import tsgm.models.augmentations

def augment_data(num_augmented=5, variance=0.01):
    """Add augmentation to raw data.
    
    Args:
        num_augmented: The number of new augmented data segments for each raw data segment
    """
    jitter_aug_model = tsgm.models.augmentations.GaussianNoise()
    
    for root, _, files  in os.walk(os.path.join(ROOT_DIR, DATA_RAW_DIR)):
        for file in files:
            src = os.path.join(root, file)            
            dest = os.path.join(ROOT_DIR, DATA_AUGMENTED_DIR, os.path.basename(root))
        
            raw_data = pd.read_csv(src)
            
            raw_data.to_csv(os.path.join(dest, file), index=False)

            augmented = raw_data.copy()
            
            # Jittering Augmentation
            samples = jitter_aug_model.generate(
                np.atleast_3d(augmented[["flow", "pressure"]].to_numpy()).swapaxes(0, 2),
                n_samples=num_augmented,
                variance=variance
            )
            
            for i in range(num_augmented):
                augmented[["flow", "pressure"]] = np.maximum(samples[i].T, 0)
                augmented.reset_index(drop=True, inplace=True)
                
                file_name = file.removesuffix(".csv") + f"-aug-{i}.csv"
                augmented.to_csv(os.path.join(dest, file_name), index=False)