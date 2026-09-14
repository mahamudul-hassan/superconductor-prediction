import pandas as pd, numpy as np, time, pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

train = pd.read_csv('/mnt/user-data/uploads/train.csv')
X_cols = [c for c in train.columns if c != 'critical_temp']
X = train[X_cols].values
y = train['critical_temp'].values
offset = 1.0
ylog = np.log(y+offset)

g = np.load('artifacts/grouped_splits.npz')
train_idx, pool_idx = g['train_idx'], g['pool_idx']
X_train, y_train, ylog_train = X[train_idx], y[train_idx], ylog[train_idx]
X_pool, y_pool, ylog_pool = X[pool_idx], y[pool_idx], ylog[pool_idx]

t0=time.time()
rf_raw_g = RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=0)
rf_raw_g.fit(X_train, y_train)
print("raw RF (grouped) fit time", time.time()-t0, flush=True)

t0=time.time()
rf_log_g = RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=0)
rf_log_g.fit(X_train, ylog_train)
print("log RF (grouped) fit time", time.time()-t0, flush=True)

# in-sample (train) performance -- overfitting check
pred_raw_train = rf_raw_g.predict(X_train)
pred_log_train = rf_log_g.predict(X_train)
print("\n--- GROUPED SPLIT (leak-free) ---")
print("Train R2 (raw):", r2_score(y_train, pred_raw_train), " MAE:", mean_absolute_error(y_train, pred_raw_train))
pred_raw_pool = rf_raw_g.predict(X_pool)
print("Pool(held-out) R2 (raw):", r2_score(y_pool, pred_raw_pool), " MAE:", mean_absolute_error(y_pool, pred_raw_pool))

pred_log_pool = rf_log_g.predict(X_pool)
pred_log_pool_orig = np.exp(pred_log_pool)-offset
pred_log_train_orig = np.exp(pred_log_train)-offset
print("Train R2 (log):", r2_score(y_train, pred_log_train_orig), " MAE:", mean_absolute_error(y_train, pred_log_train_orig))
print("Pool(held-out) R2 (log):", r2_score(y_pool, pred_log_pool_orig), " MAE:", mean_absolute_error(y_pool, pred_log_pool_orig))

with open('artifacts/models_grouped.pkl','wb') as f:
    pickle.dump({'rf_raw':rf_raw_g, 'rf_log':rf_log_g}, f)

np.savez('artifacts/pool_grouped.npz', pool_idx=pool_idx,
    y_pool=y_pool, ylog_pool=ylog_pool,
    pred_raw_pool=pred_raw_pool, pred_log_pool=pred_log_pool)

print("\nTOTAL TIME", time.time()-t0, flush=True)
