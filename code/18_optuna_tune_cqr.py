import pandas as pd, numpy as np, time, optuna
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)

train = pd.read_csv('/mnt/user-data/uploads/train.csv')
X_cols = [c for c in train.columns if c != 'critical_temp']
X = train[X_cols].values
y = train['critical_temp'].values

inner = np.load('artifacts/inner_split.npz')
train2_idx, val_idx = inner['train2_idx'], inner['val_idx']
Xt, yt = X[train2_idx], y[train2_idx]
Xv, yv = X[val_idx], y[val_idx]

def make_objective(q):
    def objective(trial):
        lr = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_iter = trial.suggest_int('max_iter', 50, 500)
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 15, 63)
        l2 = trial.suggest_float('l2_regularization', 1e-4, 10, log=True)
        min_leaf = trial.suggest_int('min_samples_leaf', 5, 100)
        m = HistGradientBoostingRegressor(loss='quantile', quantile=q, learning_rate=lr,
                max_iter=max_iter, max_leaf_nodes=max_leaf_nodes, l2_regularization=l2,
                min_samples_leaf=min_leaf, random_state=0)
        m.fit(Xt, yt)
        return mean_pinball_loss(yv, m.predict(Xv), alpha=q)
    return objective

results = {}
for q in [0.1, 0.9]:
    study = optuna.create_study(direction='minimize', study_name=f'hgb_q{q}',
                                 storage='sqlite:////home/claude/artifacts/optuna_study.db', load_if_exists=True)
    t0=time.time()
    study.optimize(make_objective(q), n_trials=60, timeout=120)
    print(f"q={q}: trials={len(study.trials)} best_val_pinball={study.best_value:.4f} time={time.time()-t0:.1f}s", flush=True)
    print("  best params:", study.best_params, flush=True)
    results[q] = study.best_params
import json
with open('artifacts/hgb_best_params.json','w') as f:
    json.dump({str(k):v for k,v in results.items()}, f, indent=2)
