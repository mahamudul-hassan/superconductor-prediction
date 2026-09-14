import pandas as pd, numpy as np, pickle, time
train = pd.read_csv('/mnt/user-data/uploads/train.csv')
uniq = pd.read_csv('/mnt/user-data/uploads/unique_m.csv')
X_cols = [c for c in train.columns if c != 'critical_temp']
X = train[X_cols].values
formula_all = uniq['material'].values

with open('artifacts/models_grouped.pkl','rb') as f:
    models = pickle.load(f)
rf_log_g = models['rf_log']

g = np.load('artifacts/pool_grouped.npz')
pool_idx = g['pool_idx']
X_pool = X[pool_idx]
formula_pool = formula_all[pool_idx]

t0=time.time()
tree_preds = np.stack([est.predict(X_pool) for est in rf_log_g.estimators_], axis=0)
std_log_pool_g = tree_preds.std(axis=0)
print("tree std computed,", time.time()-t0, "s", flush=True)

np.savez('artifacts/std_log_pool_grouped.npz', std_log_pool=std_log_pool_g, formula_pool=formula_pool)
print("unique formulas in pool:", len(set(formula_pool)), "rows:", len(formula_pool), flush=True)
