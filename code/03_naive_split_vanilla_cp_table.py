import numpy as np, json

d = np.load('artifacts/preds_seed0.npz')
y_test, y_cal = d['y_test'], d['y_cal']
ylog_test, ylog_cal = d['ylog_test'], d['ylog_cal']
pred_raw_test, pred_raw_cal = d['pred_raw_test'], d['pred_raw_cal']
pred_log_test, pred_log_cal = d['pred_log_test'], d['pred_log_cal']

t = np.load('artifacts/tree_std_seed0.npz')
std_raw_test = t['std_raw_test']
std_log_test = t['std_log_test']

offset = 1.0
alpha = 0.2  # 80% target
z = 1.2816

def summarize(lower, upper, y_true, label):
    cov = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    width = np.mean(upper - lower)
    low_mask = y_true < 10  # analog of "low-swelling regime"
    width_low = np.mean(upper[low_mask] - lower[low_mask])
    print(f"{label:25s} coverage={cov:6.2f}%  avg_width={width:7.2f}K  width(Tc<10K)={width_low:7.2f}K  n_low={low_mask.sum()}")
    return {'coverage':cov, 'avg_width':width, 'width_low':width_low}

results = {}

# 1. Standard RF heuristic (raw scale, tree std)
lower = pred_raw_test - z*std_raw_test
upper = pred_raw_test + z*std_raw_test
results['standard_rf_heuristic'] = summarize(lower, upper, y_test, "Standard RF (heuristic)")

# 2. Log-transformed RF heuristic (log scale then invert)
lower_log = pred_log_test - z*std_log_test
upper_log = pred_log_test + z*std_log_test
lower = np.exp(lower_log) - offset
upper = np.exp(upper_log) - offset
results['log_rf_heuristic'] = summarize(lower, upper, y_test, "Log-transformed RF (heuristic)")

# 3. Vanilla split-CP in log space (the paper's proposed method)
scores_cal = np.abs(ylog_cal - pred_log_cal)
n = len(scores_cal)
q_level = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
qalpha = np.quantile(scores_cal, q_level, method='higher')
lower_log = pred_log_test - qalpha
upper_log = pred_log_test + qalpha
lower = np.exp(lower_log) - offset
upper = np.exp(upper_log) - offset
results['log_cp_vanilla'] = summarize(lower, upper, y_test, "Log-transformed CP (vanilla)")
results['qalpha'] = float(qalpha)

with open('artifacts/step3_table3.json','w') as f:
    json.dump(results, f, indent=2)
