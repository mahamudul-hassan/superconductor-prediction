import pandas as pd, numpy as np, time, json
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

train = pd.read_csv('/mnt/user-data/uploads/train.csv')
X_cols = [c for c in train.columns if c != 'critical_temp']
X = train[X_cols].values
y = train['critical_temp'].values

g = np.load('artifacts/grouped_splits.npz')
train_idx, pool_idx = g['train_idx'], g['pool_idx']
X_train, y_train = X[train_idx], y[train_idx]
X_pool, y_pool = X[pool_idx], y[pool_idx]

scaler = StandardScaler().fit(X_train)
Xt_s, Xp_s = scaler.transform(X_train), scaler.transform(X_pool)

results = {}

def evaluate(name, model, use_scaled=False):
    t0 = time.time()
    Xt = Xt_s if use_scaled else X_train
    Xp = Xp_s if use_scaled else X_pool
    model.fit(Xt, y_train)
    pred = model.predict(Xp)
    r2 = r2_score(y_pool, pred)
    mae = mean_absolute_error(y_pool, pred)
    dt = time.time()-t0
    print(f"{name:25s} R2={r2:.4f}  MAE={mae:6.3f} K  fit+predict={dt:.1f}s", flush=True)
    results[name] = {'r2': r2, 'mae': mae, 'time_s': dt}

evaluate("Random Forest", RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=0))
evaluate("Extra Trees", ExtraTreesRegressor(n_estimators=200, n_jobs=1, random_state=0))
evaluate("Gradient Boosting (Hist)", HistGradientBoostingRegressor(max_iter=300, random_state=0))
evaluate("k-Nearest Neighbors", KNeighborsRegressor(n_neighbors=5, n_jobs=1), use_scaled=True)
evaluate("Ridge Regression", Ridge(alpha=1.0), use_scaled=True)
evaluate("Support Vector Regression", SVR(kernel='rbf'), use_scaled=True)

with open('artifacts/bench6_models.json','w') as f:
    json.dump(results, f, indent=2)

best = max(results.items(), key=lambda kv: kv[1]['r2'])
print("\nBEST MODEL:", best[0], best[1])
