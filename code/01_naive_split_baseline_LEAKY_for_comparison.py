import pandas as pd, numpy as np, time, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

t_start = time.time()
df = pd.read_csv('/mnt/user-data/uploads/train.csv')
df = df.drop_duplicates().reset_index(drop=True)
print("After dedup:", df.shape, flush=True)

X = df.drop(columns=['critical_temp']).values
y = df['critical_temp'].values
noe = df['number_of_elements'].values
offset = 1.0
ylog = np.log(y + offset)

seed = 0
idx = np.arange(len(y))
idx_train, idx_rest = train_test_split(idx, test_size=0.2, random_state=seed)
idx_cal, idx_test = train_test_split(idx_rest, test_size=0.5, random_state=seed)
print("train/cal/test sizes:", len(idx_train), len(idx_cal), len(idx_test), flush=True)

X_train, y_train, ylog_train = X[idx_train], y[idx_train], ylog[idx_train]
X_cal, y_cal, ylog_cal, noe_cal = X[idx_cal], y[idx_cal], ylog[idx_cal], noe[idx_cal]
X_test, y_test, ylog_test, noe_test = X[idx_test], y[idx_test], ylog[idx_test], noe[idx_test]

np.savez('artifacts/splits.npz', idx_train=idx_train, idx_cal=idx_cal, idx_test=idx_test)

print("fitting raw RF...", flush=True)
t0=time.time()
rf_raw = RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=seed)
rf_raw.fit(X_train, y_train)
print("raw RF fit time", time.time()-t0, flush=True)

print("fitting log RF...", flush=True)
t0=time.time()
rf_log = RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=seed)
rf_log.fit(X_train, ylog_train)
print("log RF fit time", time.time()-t0, flush=True)

pred_raw_test = rf_raw.predict(X_test)
pred_raw_cal = rf_raw.predict(X_cal)
r2_raw = r2_score(y_test, pred_raw_test); mae_raw = mean_absolute_error(y_test, pred_raw_test)
print("Standard RF (raw target) -> R2:", r2_raw, "MAE:", mae_raw, flush=True)

pred_log_test = rf_log.predict(X_test)
pred_log_cal = rf_log.predict(X_cal)
pred_log_test_orig = np.exp(pred_log_test) - offset
r2_log = r2_score(y_test, pred_log_test_orig); mae_log = mean_absolute_error(y_test, pred_log_test_orig)
print("Log-transformed RF -> R2:", r2_log, "MAE:", mae_log, flush=True)

import pickle
with open('artifacts/models_seed0.pkl','wb') as f:
    pickle.dump({'rf_raw':rf_raw, 'rf_log':rf_log}, f)

np.savez('artifacts/preds_seed0.npz',
    y_test=y_test, y_cal=y_cal, ylog_test=ylog_test, ylog_cal=ylog_cal,
    pred_raw_test=pred_raw_test, pred_raw_cal=pred_raw_cal,
    pred_log_test=pred_log_test, pred_log_cal=pred_log_cal,
    noe_test=noe_test, noe_cal=noe_cal)

with open('artifacts/step1_metrics.json','w') as f:
    json.dump({'r2_raw':r2_raw,'mae_raw':mae_raw,'r2_log':r2_log,'mae_log':mae_log}, f)

print("TOTAL TIME", time.time()-t_start, flush=True)
