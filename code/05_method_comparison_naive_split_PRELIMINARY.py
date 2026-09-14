import numpy as np, json, time

pool = np.load('artifacts/pool.npz')
y_pool, ylog_pool, noe_pool = pool['y_pool'], pool['ylog_pool'], pool['noe_pool']
pred_log_pool = pool['pred_log_pool']
std_log_pool = np.load('artifacts/std_log_pool.npy')
offset = 1.0
alpha = 0.20
N = len(y_pool)
eps = 0.02

def bucket(noe):
    if noe <= 2: return 'A(<=2 elem)'
    if noe == 3: return 'B(3 elem)'
    if noe == 4: return 'C(4 elem)'
    if noe == 5: return 'D(5 elem)'
    if noe == 6: return 'E(6 elem)'
    return 'F(>=7 elem)'
groups_pool = np.array([bucket(v) for v in noe_pool])
group_labels = sorted(set(groups_pool))

def conformal_q(scores, a):
    n = len(scores)
    q_level = min(np.ceil((n+1)*(1-a))/n, 1.0)
    return np.quantile(scores, q_level, method='higher')

N_REPS = 300
methods = ['vanilla_marginal', 'mondrian_unnorm', 'normalized_marginal', 'normalized_mondrian']
overall_cov = {m: [] for m in methods}
overall_width = {m: [] for m in methods}
group_cov = {m: {g: [] for g in group_labels} for m in methods}
group_width = {m: {g: [] for g in group_labels} for m in methods}

t0=time.time()
for rep in range(N_REPS):
    rng = np.random.default_rng(rep)  # SAME seeds as step5/step8
    perm = rng.permutation(N)
    half = N//2
    cal_idx, test_idx = perm[:half], perm[half:]

    ylog_cal, ylog_test = ylog_pool[cal_idx], ylog_pool[test_idx]
    predlog_cal, predlog_test = pred_log_pool[cal_idx], pred_log_pool[test_idx]
    std_cal, std_test = std_log_pool[cal_idx], std_log_pool[test_idx]
    y_test = y_pool[test_idx]
    g_cal, g_test = groups_pool[cal_idx], groups_pool[test_idx]

    raw_scores_cal = np.abs(ylog_cal - predlog_cal)
    norm_scores_cal = raw_scores_cal / (std_cal + eps)

    # --- Method 1: vanilla marginal (unnormalized, single global threshold) ---
    q = conformal_q(raw_scores_cal, alpha)
    lower = np.exp(predlog_test - q) - offset
    upper = np.exp(predlog_test + q) - offset
    overall_cov['vanilla_marginal'].append(np.mean((y_test>=lower)&(y_test<=upper))*100)
    overall_width['vanilla_marginal'].append(np.mean(upper-lower))
    for g in group_labels:
        m = (g_test==g)
        group_cov['vanilla_marginal'][g].append(np.mean((y_test[m]>=lower[m])&(y_test[m]<=upper[m]))*100)
        group_width['vanilla_marginal'][g].append(np.mean(upper[m]-lower[m]))

    # --- Method 2: Mondrian unnormalized (per-group threshold, constant width within group) ---
    lower_m = np.zeros_like(predlog_test); upper_m = np.zeros_like(predlog_test)
    for g in group_labels:
        mc, mt = (g_cal==g), (g_test==g)
        if mc.sum() < 10 or mt.sum()==0: continue
        qg = conformal_q(raw_scores_cal[mc], alpha)
        lower_m[mt] = predlog_test[mt] - qg
        upper_m[mt] = predlog_test[mt] + qg
    lower_m = np.exp(lower_m) - offset; upper_m = np.exp(upper_m) - offset
    overall_cov['mondrian_unnorm'].append(np.mean((y_test>=lower_m)&(y_test<=upper_m))*100)
    overall_width['mondrian_unnorm'].append(np.mean(upper_m-lower_m))
    for g in group_labels:
        m = (g_test==g)
        group_cov['mondrian_unnorm'][g].append(np.mean((y_test[m]>=lower_m[m])&(y_test[m]<=upper_m[m]))*100)
        group_width['mondrian_unnorm'][g].append(np.mean(upper_m[m]-lower_m[m]))

    # --- Method 3: normalized marginal (single global threshold on normalized score, scaled by local std) ---
    qn = conformal_q(norm_scores_cal, alpha)
    lower_n = np.exp(predlog_test - qn*(std_test+eps)) - offset
    upper_n = np.exp(predlog_test + qn*(std_test+eps)) - offset
    overall_cov['normalized_marginal'].append(np.mean((y_test>=lower_n)&(y_test<=upper_n))*100)
    overall_width['normalized_marginal'].append(np.mean(upper_n-lower_n))
    for g in group_labels:
        m = (g_test==g)
        group_cov['normalized_marginal'][g].append(np.mean((y_test[m]>=lower_n[m])&(y_test[m]<=upper_n[m]))*100)
        group_width['normalized_marginal'][g].append(np.mean(upper_n[m]-lower_n[m]))

    # --- Method 4: normalized + Mondrian (per-group threshold on normalized score, scaled by local std) ---
    lower_nm = np.zeros_like(predlog_test); upper_nm = np.zeros_like(predlog_test)
    for g in group_labels:
        mc, mt = (g_cal==g), (g_test==g)
        if mc.sum() < 10 or mt.sum()==0: continue
        qgn = conformal_q(norm_scores_cal[mc], alpha)
        lower_nm[mt] = predlog_test[mt] - qgn*(std_test[mt]+eps)
        upper_nm[mt] = predlog_test[mt] + qgn*(std_test[mt]+eps)
    lower_nm = np.exp(lower_nm) - offset; upper_nm = np.exp(upper_nm) - offset
    overall_cov['normalized_mondrian'].append(np.mean((y_test>=lower_nm)&(y_test<=upper_nm))*100)
    overall_width['normalized_mondrian'].append(np.mean(upper_nm-lower_nm))
    for g in group_labels:
        m = (g_test==g)
        group_cov['normalized_mondrian'][g].append(np.mean((y_test[m]>=lower_nm[m])&(y_test[m]<=upper_nm[m]))*100)
        group_width['normalized_mondrian'][g].append(np.mean(upper_nm[m]-lower_nm[m]))

print("loop time", time.time()-t0, flush=True)

summary = {}
print(f"\n{'Method':22s} {'Marginal Cov':>13s} {'Avg Width':>10s} {'Group cov range':>18s}")
for m in methods:
    oc = np.array(overall_cov[m]); ow = np.array(overall_width[m])
    gcov_means = [np.mean(group_cov[m][g]) for g in group_labels]
    spread = max(gcov_means) - min(gcov_means)
    print(f"{m:22s} {oc.mean():11.2f}%  {ow.mean():9.2f}  spread={spread:5.2f}pp (min={min(gcov_means):.1f}, max={max(gcov_means):.1f})", flush=True)
    summary[m] = {
        'marginal_coverage_mean': float(oc.mean()), 'marginal_coverage_std': float(oc.std()),
        'avg_width_mean': float(ow.mean()), 'avg_width_std': float(ow.std()),
        'group_coverage_spread_pp': float(spread),
        'per_group': {g: {'coverage_mean': float(np.mean(group_cov[m][g])),
                           'coverage_std': float(np.std(group_cov[m][g])),
                           'width_mean': float(np.mean(group_width[m][g]))} for g in group_labels}
    }

with open('artifacts/step10_compare4.json','w') as f:
    json.dump(summary, f, indent=2)
