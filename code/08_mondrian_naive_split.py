import numpy as np, json, time

d = np.load('artifacts/pool.npz')
y_pool, ylog_pool, noe_pool = d['y_pool'], d['ylog_pool'], d['noe_pool']
pred_log_pool = d['pred_log_pool']
offset = 1.0
N = len(y_pool)

# bucket number_of_elements into 6 groups to keep enough samples per group
def bucket(noe):
    if noe <= 2: return 'A(<=2 elem)'
    if noe == 3: return 'B(3 elem)'
    if noe == 4: return 'C(4 elem)'
    if noe == 5: return 'D(5 elem)'
    if noe == 6: return 'E(6 elem)'
    return 'F(>=7 elem)'

groups_pool = np.array([bucket(v) for v in noe_pool])
group_labels = sorted(set(groups_pool), key=lambda s: s)
print("pool group counts:", {g:int((groups_pool==g).sum()) for g in group_labels}, flush=True)

alpha = 0.20  # 80% target
def conformal_q(scores, a):
    n = len(scores)
    q_level = min(np.ceil((n+1)*(1-a))/n, 1.0)
    return np.quantile(scores, q_level, method='higher')

N_REPS = 300
marginal_cov = {g: [] for g in group_labels}
mondrian_cov = {g: [] for g in group_labels}
marginal_width = {g: [] for g in group_labels}
mondrian_width = {g: [] for g in group_labels}
group_test_n = {g: [] for g in group_labels}

t0=time.time()
for rep in range(N_REPS):
    rng = np.random.default_rng(1000+rep)
    perm = rng.permutation(N)
    half = N//2
    cal_idx, test_idx = perm[:half], perm[half:]

    ylog_cal, ylog_test = ylog_pool[cal_idx], ylog_pool[test_idx]
    predlog_cal, predlog_test = pred_log_pool[cal_idx], pred_log_pool[test_idx]
    y_test = y_pool[test_idx]
    g_cal, g_test = groups_pool[cal_idx], groups_pool[test_idx]

    scores_cal = np.abs(ylog_cal - predlog_cal)
    # marginal (global) threshold
    q_global = conformal_q(scores_cal, alpha)
    lower_g = np.exp(predlog_test - q_global) - offset
    upper_g = np.exp(predlog_test + q_global) - offset

    for g in group_labels:
        mask_test = (g_test == g)
        if mask_test.sum() == 0:
            continue
        # marginal CP evaluated on this subgroup
        cov_m = np.mean((y_test[mask_test]>=lower_g[mask_test]) & (y_test[mask_test]<=upper_g[mask_test]))*100
        width_m = np.mean(upper_g[mask_test]-lower_g[mask_test])
        marginal_cov[g].append(cov_m)
        marginal_width[g].append(width_m)
        group_test_n[g].append(int(mask_test.sum()))

        # Mondrian: calibrate using only this group's calibration residuals
        mask_cal = (g_cal == g)
        if mask_cal.sum() < 10:
            continue
        q_g = conformal_q(scores_cal[mask_cal], alpha)
        lower_mon = np.exp(predlog_test[mask_test] - q_g) - offset
        upper_mon = np.exp(predlog_test[mask_test] + q_g) - offset
        cov_mon = np.mean((y_test[mask_test]>=lower_mon) & (y_test[mask_test]<=upper_mon))*100
        width_mon = np.mean(upper_mon-lower_mon)
        mondrian_cov[g].append(cov_mon)
        mondrian_width[g].append(width_mon)

print("mondrian loop time", time.time()-t0, flush=True)

summary = {}
for g in group_labels:
    mc, mw = np.array(marginal_cov[g]), np.array(marginal_width[g])
    dc, dw = np.array(mondrian_cov[g]), np.array(mondrian_width[g])
    n_avg = np.mean(group_test_n[g])
    summary[g] = {
        'avg_test_n': float(n_avg),
        'marginal_coverage_mean': float(mc.mean()), 'marginal_coverage_std': float(mc.std()),
        'mondrian_coverage_mean': float(dc.mean()) if len(dc) else None,
        'mondrian_coverage_std': float(dc.std()) if len(dc) else None,
        'marginal_width_mean': float(mw.mean()),
        'mondrian_width_mean': float(dw.mean()) if len(dw) else None,
    }
    print(f"{g:15s} n~{n_avg:6.1f} | marginal cov={mc.mean():6.2f}+/-{mc.std():4.2f} width={mw.mean():6.2f} | mondrian cov={dc.mean() if len(dc) else float('nan'):6.2f}+/-{dc.std() if len(dc) else float('nan'):4.2f} width={dw.mean() if len(dw) else float('nan'):6.2f}", flush=True)

with open('artifacts/step6_mondrian.json','w') as f:
    json.dump(summary, f, indent=2)
