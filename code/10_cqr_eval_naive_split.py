import numpy as np, json, time

d = np.load('artifacts/pool.npz')
y_pool, noe_pool = d['y_pool'], d['noe_pool']
cqr = np.load('artifacts/cqr_pool_preds.npz')
N = len(y_pool)
alphas = [0.30, 0.20, 0.10, 0.05]

def conformal_q(scores, a):
    n = len(scores)
    q_level = min(np.ceil((n+1)*(1-a))/n, 1.0)
    return np.quantile(scores, q_level, method='higher')

def bucket(noe):
    if noe <= 2: return 'A(<=2 elem)'
    if noe == 3: return 'B(3 elem)'
    if noe == 4: return 'C(4 elem)'
    if noe == 5: return 'D(5 elem)'
    if noe == 6: return 'E(6 elem)'
    return 'F(>=7 elem)'
groups_pool = np.array([bucket(v) for v in noe_pool])
group_labels = sorted(set(groups_pool))

N_REPS = 300
cov_results = {a: [] for a in alphas}
width_results = {a: [] for a in alphas}
group_cov_02 = {g: [] for g in group_labels}   # CQR per-group coverage at alpha=0.2, marginal calibration

t0=time.time()
for rep in range(N_REPS):
    rng = np.random.default_rng(rep)  # SAME seeds as step5 for fair comparison
    perm = rng.permutation(N)
    half = N//2
    cal_idx, test_idx = perm[:half], perm[half:]
    y_test = y_pool[test_idx]
    g_test = groups_pool[test_idx]

    for a in alphas:
        lo_pool, hi_pool = cqr[f'lo_{a}'], cqr[f'hi_{a}']
        lo_cal, hi_cal = lo_pool[cal_idx], hi_pool[cal_idx]
        lo_test, hi_test = lo_pool[test_idx], hi_pool[test_idx]
        y_cal = y_pool[cal_idx]
        scores_cal = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
        q = conformal_q(scores_cal, a)
        lower = lo_test - q
        upper = hi_test + q
        cov = np.mean((y_test>=lower)&(y_test<=upper))*100
        width = np.mean(upper-lower)
        cov_results[a].append(cov)
        width_results[a].append(width)

        if a == 0.20:
            for g in group_labels:
                m = (g_test==g)
                if m.sum()==0: continue
                covg = np.mean((y_test[m]>=lower[m])&(y_test[m]<=upper[m]))*100
                group_cov_02[g].append(covg)

print("time", time.time()-t0, flush=True)

summary = {}
for a in alphas:
    covs, ws = np.array(cov_results[a]), np.array(width_results[a])
    summary[str(int((1-a)*100))+"%"] = {'mean_coverage': float(covs.mean()), 'std_coverage': float(covs.std()),
                                          'mean_width': float(ws.mean()), 'std_width': float(ws.std())}
    print(f"CQR target {(1-a)*100:.0f}%: coverage={covs.mean():.2f}+/-{covs.std():.2f}, width={ws.mean():.2f}+/-{ws.std():.2f}", flush=True)

group_summary = {}
for g in group_labels:
    c = np.array(group_cov_02[g])
    group_summary[g] = {'mean_coverage': float(c.mean()), 'std_coverage': float(c.std())}
    print(f"CQR group {g:15s} coverage(alpha=0.2) = {c.mean():.2f}+/-{c.std():.2f}", flush=True)

with open('artifacts/step8_cqr.json','w') as f:
    json.dump({'reliability':summary, 'group_coverage_at_80':group_summary}, f, indent=2)
