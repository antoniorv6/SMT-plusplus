import os
import numpy as np
from sklearn.model_selection import KFold, train_test_split

os.makedirs('partitions_polish_scores/excerpts', exist_ok=True)

all_files = []
for file in os.listdir('polish_scores_dataset'):
    if file.endswith('.ekern'):
        all_files.append(file)

np.random.seed(42)
np.random.shuffle(all_files)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_index, test_index) in enumerate(kf.split(all_files)):
    train_files = ["polish_scores_dataset/" + all_files[i] for i in train_index]
    test_files = ["polish_scores_dataset/" + all_files[i] for i in test_index]
    
    os.makedirs('partitions_polish_scores/excerpts/fold_' + str(i), exist_ok=True)
    train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)
    
    with open(f'partitions_polish_scores/excerpts/fold_{i}/train.txt', 'w') as f:
        f.write('\n'.join(train_files))
    with open(f'partitions_polish_scores/excerpts/fold_{i}/test.txt', 'w') as f:
        f.write('\n'.join(test_files))
    with open(f'partitions_polish_scores/excerpts/fold_{i}/val.txt', 'w') as f:
        f.write('\n'.join(val_files))
    
    
