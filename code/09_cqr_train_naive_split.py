import pandas as pd, numpy as np, time, pickle
from sklearn.ensemble import HistGradientBoostingRegressor

df = pd.read_csv('/mnt/user-data/uploads/train.csv').drop_duplicates().reset_index(drop=True)
X = df.drop(columns=['critical_temp']).values
y = df['critical_temp'].values

splits = np.load('artifacts/splits.npz')
idx_train, idx_cal, idx_test = splits['idx_train'], splits['idx_cal'], splits['idx_test']
X_train, y_train = X[idx_train], y[idx_train]

alphas = [0.30, 0.20, 0.10, 0.05]
models = {}
t0=time.time()
for a in alphas:
    qlo, qhi = a/2, 1-a/2
    m_lo = HistGradientBoostingRegressor(loss='quantile', quantile=qlo, max_iter=200, random_state=0)
    m_hi = HistGradientBoostingRegressor(loss='quantile', quantile=qhi, max_iter=200, random_state=0)
    m_lo.fit(X_train, y_train)
    m_hi.fit(X_train, y_train)
    models[a] = (m_lo, m_hi)
    print(f"alpha={a} trained, qlo={qlo}, qhi={qhi}, elapsed={time.time()-t0:.1f}s", flush=True)

with open('artifacts/cqr_models.pkl','wb') as f:
    pickle.dump(models, f)

# predict on full pool (cal+test combined, fixed hold-out set) for cheap repeated-split reuse
idx_rest = np.concatenate([idx_cal, idx_test])
X_pool = X[idx_rest]
preds = {}
for a in alphas:
    m_lo, m_hi = models[a]
    preds[a] = (m_lo.predict(X_pool), m_hi.predict(X_pool))
    print(f"alpha={a} predicted on pool", flush=True)

np.savez('artifacts/cqr_pool_preds.npz', idx_rest=idx_rest,
    **{f'lo_{a}': preds[a][0] for a in alphas},
    **{f'hi_{a}': preds[a][1] for a in alphas})
print("TOTAL TIME", time.time()-t0, flush=True)
