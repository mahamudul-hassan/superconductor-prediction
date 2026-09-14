import pandas as pd, numpy as np, pickle, time
train = pd.read_csv('/mnt/user-data/uploads/train.csv')
uniq = pd.read_csv('/mnt/user-data/uploads/unique_m.csv')
X_cols = [c for c in train.columns if c != 'critical_temp']
X = train[X_cols].values
formula_all = uniq['material'].values

with open('artifacts/et_models_tuned.pkl','rb') as f:
    models = pickle.load(f)
et_log = models['et_log']

pt = np.load('artifacts/et_pool_tuned.npz')
pool_idx = pt['pool_idx']
X_pool = X[pool_idx]
formula_pool = formula_all[pool_idx]

t0=time.time()
tree_preds = np.stack([est.predict(X_pool) for est in et_log.estimators_], axis=0)
std_log_pool = tree_preds.std(axis=0)
print("tree std computed", time.time()-t0, "s", flush=True)
np.savez('artifacts/et_std_log_pool.npz', std_log_pool=std_log_pool, formula_pool=formula_pool)
