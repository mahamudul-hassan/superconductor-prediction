import pandas as pd, numpy as np, time, json

train = pd.read_csv('/mnt/user-data/uploads/train.csv')
noe_all = train['number_of_elements'].values

g = np.load('artifacts/grouped_splits.npz')
pool_idx = g['pool_idx']
noe_pool = noe_all[pool_idx]

pt = np.load('artifacts/et_pool_tuned.npz')
y_pool, ylog_pool = pt['y_pool'], pt['ylog_pool']
pred_log_pool = pt['pred_log_pool']

std_data = np.load('artifacts/et_std_log_pool.npz', allow_pickle=True)
std_log_pool = std_data['std_log_pool']
formula_pool = std_data['formula_pool']

cqr = np.load('artifacts/cqr_pool_tuned.npz')
lo_pool, hi_pool = cqr['lo_pool'], cqr['hi_pool']

offset = 1.0
alpha = 0.20
eps = 0.1
N = len(y_pool)

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

unique_formulas = np.array(sorted(set(formula_pool)))
formula_to_rows = {}
for i, f in enumerate(formula_pool):
    formula_to_rows.setdefault(f, []).append(i)

N_REPS = 300
methods = ['vanilla_marginal', 'mondrian_unnorm', 'normalized_marginal', 'normalized_mondrian', 'cqr_marginal']
overall_cov = {m: [] for m in methods}
overall_width = {m: [] for m in methods}
group_cov = {m: {g: [] for g in group_labels} for m in methods}
group_width = {m: {g: [] for g in group_labels} for m in methods}

t0 = time.time()
for rep in range(N_REPS):
    rng = np.random.default_rng(rep)
    perm_formulas = rng.permutation(unique_formulas)
    half = len(perm_formulas)//2
    cal_formulas, test_formulas = set(perm_formulas[:half]), set(perm_formulas[half:])
    cal_idx = np.array([i for f in cal_formulas for i in formula_to_rows[f]])
    test_idx = np.array([i for f in test_formulas for i in formula_to_rows[f]])

    ylog_cal, ylog_test = ylog_pool[cal_idx], ylog_pool[test_idx]
    predlog_cal, predlog_test = pred_log_pool[cal_idx], pred_log_pool[test_idx]
    std_cal, std_test = std_log_pool[cal_idx], std_log_pool[test_idx]
    y_test, y_cal = y_pool[test_idx], y_pool[cal_idx]
    g_cal, g_test = groups_pool[cal_idx], groups_pool[test_idx]

    raw_scores_cal = np.abs(ylog_cal - predlog_cal)
    norm_scores_cal = raw_scores_cal / (std_cal + eps)

    q = conformal_q(raw_scores_cal, alpha)
    lower = np.exp(predlog_test - q) - offset
    upper = np.exp(predlog_test + q) - offset
    overall_cov['vanilla_marginal'].append(np.mean((y_test>=lower)&(y_test<=upper))*100)
    overall_width['vanilla_marginal'].append(np.mean(upper-lower))
    for gl in group_labels:
        m = (g_test==gl)
        if m.sum()==0: continue
        group_cov['vanilla_marginal'][gl].append(np.mean((y_test[m]>=lower[m])&(y_test[m]<=upper[m]))*100)
        group_width['vanilla_marginal'][gl].append(np.mean(upper[m]-lower[m]))

    lower_m = np.zeros_like(predlog_test); upper_m = np.zeros_like(predlog_test)
    for gl in group_labels:
        mc, mt = (g_cal==gl), (g_test==gl)
        if mc.sum() < 10 or mt.sum()==0: continue
        qg = conformal_q(raw_scores_cal[mc], alpha)
        lower_m[mt] = predlog_test[mt] - qg
        upper_m[mt] = predlog_test[mt] + qg
    lower_m = np.exp(lower_m)-offset; upper_m = np.exp(upper_m)-offset
    overall_cov['mondrian_unnorm'].append(np.mean((y_test>=lower_m)&(y_test<=upper_m))*100)
    overall_width['mondrian_unnorm'].append(np.mean(upper_m-lower_m))
    for gl in group_labels:
        m = (g_test==gl)
        if m.sum()==0: continue
        group_cov['mondrian_unnorm'][gl].append(np.mean((y_test[m]>=lower_m[m])&(y_test[m]<=upper_m[m]))*100)
        group_width['mondrian_unnorm'][gl].append(np.mean(upper_m[m]-lower_m[m]))

    qn = conformal_q(norm_scores_cal, alpha)
    lower_n = np.exp(predlog_test - qn*(std_test+eps)) - offset
    upper_n = np.exp(predlog_test + qn*(std_test+eps)) - offset
    overall_cov['normalized_marginal'].append(np.mean((y_test>=lower_n)&(y_test<=upper_n))*100)
    overall_width['normalized_marginal'].append(np.mean(upper_n-lower_n))
    for gl in group_labels:
        m = (g_test==gl)
        if m.sum()==0: continue
        group_cov['normalized_marginal'][gl].append(np.mean((y_test[m]>=lower_n[m])&(y_test[m]<=upper_n[m]))*100)
        group_width['normalized_marginal'][gl].append(np.mean(upper_n[m]-lower_n[m]))

    lower_nm = np.zeros_like(predlog_test); upper_nm = np.zeros_like(predlog_test)
    for gl in group_labels:
        mc, mt = (g_cal==gl), (g_test==gl)
        if mc.sum() < 10 or mt.sum()==0: continue
        qgn = conformal_q(norm_scores_cal[mc], alpha)
        lower_nm[mt] = predlog_test[mt] - qgn*(std_test[mt]+eps)
        upper_nm[mt] = predlog_test[mt] + qgn*(std_test[mt]+eps)
    lower_nm = np.exp(lower_nm)-offset; upper_nm = np.exp(upper_nm)-offset
    overall_cov['normalized_mondrian'].append(np.mean((y_test>=lower_nm)&(y_test<=upper_nm))*100)
    overall_width['normalized_mondrian'].append(np.mean(upper_nm-lower_nm))
    for gl in group_labels:
        m = (g_test==gl)
        if m.sum()==0: continue
        group_cov['normalized_mondrian'][gl].append(np.mean((y_test[m]>=lower_nm[m])&(y_test[m]<=upper_nm[m]))*100)
        group_width['normalized_mondrian'][gl].append(np.mean(upper_nm[m]-lower_nm[m]))

    lo_cal, hi_cal = lo_pool[cal_idx], hi_pool[cal_idx]
    lo_test, hi_test = lo_pool[test_idx], hi_pool[test_idx]
    scores_cqr_cal = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    qc = conformal_q(scores_cqr_cal, alpha)
    lower_c = lo_test - qc; upper_c = hi_test + qc
    overall_cov['cqr_marginal'].append(np.mean((y_test>=lower_c)&(y_test<=upper_c))*100)
    overall_width['cqr_marginal'].append(np.mean(upper_c-lower_c))
    for gl in group_labels:
        m = (g_test==gl)
        if m.sum()==0: continue
        group_cov['cqr_marginal'][gl].append(np.mean((y_test[m]>=lower_c[m])&(y_test[m]<=upper_c[m]))*100)
        group_width['cqr_marginal'][gl].append(np.mean(upper_c[m]-lower_c[m]))

print("loop time", time.time()-t0, flush=True)
print(f"\n{'Method':22s} {'Marginal Cov':>13s} {'Avg Width':>10s} {'Group spread':>16s}")
summary = {}
for m in methods:
    oc, ow = np.array(overall_cov[m]), np.array(overall_width[m])
    gmeans = [np.mean(group_cov[m][gl]) for gl in group_labels if len(group_cov[m][gl])>0]
    spread = max(gmeans)-min(gmeans)
    print(f"{m:22s} {oc.mean():11.2f}%  {ow.mean():9.2f}  {spread:14.2f}pp (min={min(gmeans):.1f} max={max(gmeans):.1f})", flush=True)
    summary[m] = {'marginal_coverage_mean': float(oc.mean()), 'avg_width_mean': float(ow.mean()),
                  'group_spread_pp': float(spread),
                  'per_group': {gl: {'cov': float(np.mean(group_cov[m][gl])),
                                      'width': float(np.mean(group_width[m][gl]))} for gl in group_labels}}
with open('artifacts/et_final_eval.json','w') as f:
    json.dump(summary, f, indent=2)
