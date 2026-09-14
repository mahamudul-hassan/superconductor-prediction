import pandas as pd, numpy as np, time, optuna
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

train = pd.read_csv('/mnt/user-data/uploads/train.csv')
X_cols = [c for c in train.columns if c != 'critical_temp']
X = train[X_cols].values
y = train['critical_temp'].values
offset = 1.0
ylog = np.log(y+offset)

inner = np.load('artifacts/inner_split.npz')
train2_idx, val_idx = inner['train2_idx'], inner['val_idx']
Xt, ylogt = X[train2_idx], ylog[train2_idx]
Xv, yv = X[val_idx], y[val_idx]

def objective(trial):
    max_depth = trial.suggest_categorical('max_depth', [8, 12, 16, 20, 25, 30, None])
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0])
    et = ExtraTreesRegressor(n_estimators=30, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
            max_features=max_features, n_jobs=1, random_state=0)
    et.fit(Xt, ylogt)
    pred_val = np.exp(et.predict(Xv)) - offset
    return mean_absolute_error(yv, pred_val)

study = optuna.create_study(direction='minimize', study_name='et_tuning',
                             storage='sqlite:////home/claude/artifacts/optuna_study.db', load_if_exists=True)
t0 = time.time()
study.optimize(objective, n_trials=200, timeout=250)
print(f"trials: {len(study.trials)}, elapsed: {time.time()-t0:.1f}s", flush=True)
print("best value (val MAE):", study.best_value, flush=True)
print("best params:", study.best_params, flush=True)
