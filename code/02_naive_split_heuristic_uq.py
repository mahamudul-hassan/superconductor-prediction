import pandas as pd, numpy as np, time, pickle, json
t0=time.time()
with open('artifacts/models_seed0.pkl','rb') as f:
    models = pickle.load(f)
rf_raw, rf_log = models['rf_raw'], models['rf_log']
print("model load time", time.time()-t0, flush=True)

d = np.load('artifacts/preds_seed0.npz')
y_test, y_cal = d['y_test'], d['y_cal']
ylog_test, ylog_cal = d['ylog_test'], d['ylog_cal']
pred_raw_test, pred_raw_cal = d['pred_raw_test'], d['pred_raw_cal']
pred_log_test, pred_log_cal = d['pred_log_test'], d['pred_log_cal']
noe_test, noe_cal = d['noe_test'], d['noe_cal']

# reload X for tree-level predictions
Xall = pd.read_csv('/mnt/user-data/uploads/train.csv').drop_duplicates().reset_index(drop=True).drop(columns=['critical_temp']).values
splits = np.load('artifacts/splits.npz')
idx_cal, idx_test = splits['idx_cal'], splits['idx_test']
X_cal, X_test = Xall[idx_cal], Xall[idx_test]

offset = 1.0

def tree_std(rf, X):
    t0=time.time()
    preds = np.stack([est.predict(X) for est in rf.estimators_], axis=0)
    std = preds.std(axis=0)
    print("  tree_std time", time.time()-t0, "shape", preds.shape, flush=True)
    return std

print("computing tree std for raw RF (cal+test)...", flush=True)
std_raw_cal = tree_std(rf_raw, X_cal)
std_raw_test = tree_std(rf_raw, X_test)

print("computing tree std for log RF (cal+test)...", flush=True)
std_log_cal = tree_std(rf_log, X_cal)
std_log_test = tree_std(rf_log, X_test)

np.savez('artifacts/tree_std_seed0.npz',
    std_raw_cal=std_raw_cal, std_raw_test=std_raw_test,
    std_log_cal=std_log_cal, std_log_test=std_log_test)

print("TOTAL TIME", time.time()-t0, flush=True)
