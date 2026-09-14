import pandas as pd, numpy as np, time, json
from sklearn.ensemble import HistGradientBoostingRegressor

train = pd.read_csv('/mnt/user-data/uploads/train.csv')
X_cols = [c for c in train.columns if c != 'critical_temp']
X = train[X_cols].values
y = train['critical_temp'].values

g = np.load('artifacts/grouped_splits.npz')
train_idx, pool_idx = g['train_idx'], g['pool_idx']
X_train, y_train = X[train_idx], y[train_idx]
X_pool = X[pool_idx]

best = json.load(open('artifacts/hgb_best_params.json'))

t0=time.time()
m_lo = HistGradientBoostingRegressor(loss='quantile', quantile=0.10, random_state=0, **best['0.1'])
m_hi = HistGradientBoostingRegressor(loss='quantile', quantile=0.90, random_state=0, **best['0.9'])
m_lo.fit(X_train, y_train)
m_hi.fit(X_train, y_train)
print("tuned CQR fit time", time.time()-t0, flush=True)

lo_pool = m_lo.predict(X_pool)
hi_pool = m_hi.predict(X_pool)
np.savez('artifacts/cqr_pool_tuned.npz', lo_pool=lo_pool, hi_pool=hi_pool)
print("saved", flush=True)
