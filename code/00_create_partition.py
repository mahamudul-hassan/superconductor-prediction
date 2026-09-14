"""
Create the train / held-out-pool partition used throughout the analysis.

Compounds are grouped by their underlying chemical formula (from unique_m.csv)
so that no formula appears in more than one partition -- see Methods,
"Dataset", in the manuscript. This produces artifacts/grouped_splits.npz,
which every downstream script (partition-dependent training, tuning,
conformal-prediction evaluation) reads.
"""
import pandas as pd, numpy as np
from sklearn.model_selection import GroupShuffleSplit

train = pd.read_csv('/mnt/user-data/uploads/train.csv')
uniq = pd.read_csv('/mnt/user-data/uploads/unique_m.csv')
formula = uniq['material'].values

X_cols = [c for c in train.columns if c != 'critical_temp']
X = train[X_cols].values
y = train['critical_temp'].values

# 80% train / 20% held-out pool, grouped by formula
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, pool_idx = next(gss1.split(X, y, groups=formula))
print("train rows:", len(train_idx), " pool rows:", len(pool_idx))
print("unique formulas -> train:", len(set(formula[train_idx])),
      " pool:", len(set(formula[pool_idx])),
      " overlap:", len(set(formula[train_idx]) & set(formula[pool_idx])))

np.savez('artifacts/grouped_splits.npz', train_idx=train_idx, pool_idx=pool_idx)

# Inner split of the training set only (train2 / val), for leak-safe
# hyperparameter tuning -- the held-out pool above is never touched here.
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
formula_train = formula[train_idx]
t2_sub, val_sub = next(gss2.split(X[train_idx], y[train_idx], groups=formula_train))
train2_idx = train_idx[t2_sub]
val_idx = train_idx[val_sub]
print("train2 rows:", len(train2_idx), " val rows:", len(val_idx))
np.savez('artifacts/inner_split.npz', train2_idx=train2_idx, val_idx=val_idx)
