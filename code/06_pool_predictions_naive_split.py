import pandas as pd, numpy as np, time, pickle
t0=time.time()
with open('artifacts/models_seed0.pkl','rb') as f:
    models = pickle.load(f)
rf_raw, rf_log = models['rf_raw'], models['rf_log']
print("model load", time.time()-t0, flush=True)

df = pd.read_csv('/mnt/user-data/uploads/train.csv').drop_duplicates().reset_index(drop=True)
X = df.drop(columns=['critical_temp']).values
y = df['critical_temp'].values
noe = df['number_of_elements'].values
offset = 1.0
ylog = np.log(y+offset)

splits = np.load('artifacts/splits.npz')
idx_cal, idx_test = splits['idx_cal'], splits['idx_test']
idx_rest = np.concatenate([idx_cal, idx_test])  # fixed 20% hold-out pool, ~4240 rows
print("pool size:", len(idx_rest), flush=True)

X_pool = X[idx_rest]
y_pool = y[idx_rest]
ylog_pool = ylog[idx_rest]
noe_pool = noe[idx_rest]

t0=time.time()
pred_raw_pool = rf_raw.predict(X_pool)
pred_log_pool = rf_log.predict(X_pool)
print("predict time", time.time()-t0, flush=True)

np.savez('artifacts/pool.npz',
    idx_rest=idx_rest, y_pool=y_pool, ylog_pool=ylog_pool, noe_pool=noe_pool,
    pred_raw_pool=pred_raw_pool, pred_log_pool=pred_log_pool)
print("saved pool.npz", flush=True)
