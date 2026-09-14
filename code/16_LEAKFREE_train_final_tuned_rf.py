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

BEST = dict(max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features='sqrt')

t0=time.time()
rf_raw_t = RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=0, **BEST)
rf_raw_t.fit(X_train, y_train)
print("raw RF (tuned) fit time", time.time()-t0, flush=True)

t0=time.time()
rf_log_t = RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=0, **BEST)
rf_log_t.fit(X_train, ylog_train)
print("log RF (tuned) fit time", time.time()-t0, flush=True)

pred_raw_train = rf_raw_t.predict(X_train)
pred_raw_pool = rf_raw_t.predict(X_pool)
pred_log_train = rf_log_t.predict(X_train)
pred_log_pool = rf_log_t.predict(X_pool)
pred_log_train_orig = np.exp(pred_log_train)-offset
pred_log_pool_orig = np.exp(pred_log_pool)-offset

print("\n--- TUNED MODEL (leak-free split, Optuna-selected hyperparams, n_estimators=200) ---")
print(f"Raw:  train R2={r2_score(y_train,pred_raw_train):.4f} MAE={mean_absolute_error(y_train,pred_raw_train):.3f}  |  held-out R2={r2_score(y_pool,pred_raw_pool):.4f} MAE={mean_absolute_error(y_pool,pred_raw_pool):.3f}")
print(f"Log:  train R2={r2_score(y_train,pred_log_train_orig):.4f} MAE={mean_absolute_error(y_train,pred_log_train_orig):.3f}  |  held-out R2={r2_score(y_pool,pred_log_pool_orig):.4f} MAE={mean_absolute_error(y_pool,pred_log_pool_orig):.3f}")

with open('artifacts/models_tuned.pkl','wb') as f:
    pickle.dump({'rf_raw':rf_raw_t, 'rf_log':rf_log_t}, f)
np.savez('artifacts/pool_tuned.npz', pool_idx=pool_idx, y_pool=y_pool, ylog_pool=ylog_pool,
    pred_raw_pool=pred_raw_pool, pred_log_pool=pred_log_pool)
print("saved", flush=True)
