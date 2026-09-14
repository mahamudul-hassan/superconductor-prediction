import numpy as np, json, time
rng_master = np.random.default_rng(123)

d = np.load('artifacts/pool.npz')
y_pool, ylog_pool, noe_pool = d['y_pool'], d['ylog_pool'], d['noe_pool']
pred_raw_pool, pred_log_pool = d['pred_raw_pool'], d['pred_log_pool']
offset = 1.0
N = len(y_pool)
print("pool N =", N, flush=True)

def conformal_q(scores, alpha):
    n = len(scores)
    q_level = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    return np.quantile(scores, q_level, method='higher')

# ---- Part A: repeated-split reliability curve for vanilla log-CP across alpha levels ----
alphas = [0.30, 0.20, 0.10, 0.05]
N_REPS = 300
cov_results = {a: [] for a in alphas}
width_results = {a: [] for a in alphas}

t0=time.time()
for rep in range(N_REPS):
    rng = np.random.default_rng(rep)
    perm = rng.permutation(N)
    half = N//2
    cal_idx, test_idx = perm[:half], perm[half:]
    ylog_cal, ylog_test = ylog_pool[cal_idx], ylog_pool[test_idx]
    predlog_cal, predlog_test = pred_log_pool[cal_idx], pred_log_pool[test_idx]
    y_test = y_pool[test_idx]
    scores_cal = np.abs(ylog_cal - predlog_cal)
    for a in alphas:
        q = conformal_q(scores_cal, a)
        lower = np.exp(predlog_test - q) - offset
        upper = np.exp(predlog_test + q) - offset
        cov = np.mean((y_test>=lower)&(y_test<=upper))*100
        width = np.mean(upper-lower)
        cov_results[a].append(cov)
        width_results[a].append(width)
print("repeated split time", time.time()-t0, flush=True)

summary = {}
for a in alphas:
    covs = np.array(cov_results[a]); ws = np.array(width_results[a])
    summary[str(int((1-a)*100))+"%"] = {
        'target': (1-a)*100,
        'mean_coverage': float(covs.mean()), 'std_coverage': float(covs.std()),
        'min_coverage': float(covs.min()), 'max_coverage': float(covs.max()),
        'mean_width': float(ws.mean()), 'std_width': float(ws.std())
    }
    print(f"target {(1-a)*100:.0f}%: coverage = {covs.mean():.2f} +/- {covs.std():.2f} (range {covs.min():.2f}-{covs.max():.2f}), width = {ws.mean():.2f} +/- {ws.std():.2f}", flush=True)

with open('artifacts/step5_reliability.json','w') as f:
    json.dump(summary, f, indent=2)
np.savez('artifacts/step5_raw_reps.npz', **{f'cov_{a}':np.array(cov_results[a]) for a in alphas},
          **{f'width_{a}':np.array(width_results[a]) for a in alphas})
