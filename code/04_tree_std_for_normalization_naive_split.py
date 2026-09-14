import pandas as pd, numpy as np, pickle, time, json

t0=time.time()
with open('artifacts/models_seed0.pkl','rb') as f:
    models = pickle.load(f)
rf_log = models['rf_log']

df = pd.read_csv('/mnt/user-data/uploads/train.csv').drop_duplicates().reset_index(drop=True)
X = df.drop(columns=['critical_temp']).values

pool = np.load('artifacts/pool.npz')
idx_rest = pool['idx_rest']
y_pool, ylog_pool, noe_pool = pool['y_pool'], pool['ylog_pool'], pool['noe_pool']
pred_log_pool = pool['pred_log_pool']
X_pool = X[idx_rest]

print("computing tree-level std on full pool (for local normalization)...", flush=True)
tree_preds = np.stack([est.predict(X_pool) for est in rf_log.estimators_], axis=0)  # (100, 4240) in log space
std_log_pool = tree_preds.std(axis=0)
print("done,", time.time()-t0, "s. std range:", std_log_pool.min(), std_log_pool.max(), flush=True)

np.save('artifacts/std_log_pool.npy', std_log_pool)
