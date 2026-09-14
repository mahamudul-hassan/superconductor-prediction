import numpy as np, json, time

train_noe = None
import pandas as pd
train = pd.read_csv('/mnt/user-data/uploads/train.csv')
noe_all = train['number_of_elements'].values

g = np.load('artifacts/grouped_splits.npz')
pool_idx = g['pool_idx']
noe_pool = noe_all[pool_idx]

pt = np.load('artifacts/pool_tuned.npz')
y_pool, ylog_pool = pt['y_pool'], pt['ylog_pool']
pred_log_pool = pt['pred_log_pool']

std_data = np.load('artifacts/std_log_pool_tuned.npz', allow_pickle=True)
std_log_pool = std_data['std_log_pool']
formula_pool = std_data['formula_pool']

offset = 1.0
eps = 0.02
alphas = [0.30, 0.20, 0.10, 0.05]
N = len(y_pool)

def conformal_q(scores, a):
    n = len(scores)
    q_level = min(np.ceil((n+1)*(1-a))/n, 1.0)
    return np.quantile(scores, q_level, method='higher')

def bucket(noe):
    if noe <= 2: return 'A'
    if noe == 3: return 'B'
    if noe == 4: return 'C'
    if noe == 5: return 'D'
    if noe == 6: return 'E'
    return 'F'
groups_pool = np.array([bucket(v) for v in noe_pool])
group_labels = sorted(set(groups_pool))

unique_formulas = np.array(sorted(set(formula_pool)))
formula_to_rows = {}
for i, f in enumerate(formula_pool):
    formula_to_rows.setdefault(f, []).append(i)

N_REPS = 300
cov_vanilla = {a: [] for a in alphas}
width_vanilla = {a: [] for a in alphas}
cov_combined = {a: [] for a in alphas}
width_combined = {a: [] for a in alphas}

t0=time.time()
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
    y_test = y_pool[test_idx]
    g_cal, g_test = groups_pool[cal_idx], groups_pool[test_idx]

    raw_scores_cal = np.abs(ylog_cal - predlog_cal)
    norm_scores_cal = raw_scores_cal / (std_cal + eps)

    for a in alphas:
        q = conformal_q(raw_scores_cal, a)
        lower = np.exp(predlog_test - q) - offset
        upper = np.exp(predlog_test + q) - offset
        cov_vanilla[a].append(np.mean((y_test>=lower)&(y_test<=upper))*100)
        width_vanilla[a].append(np.mean(upper-lower))

        lower_c = np.zeros_like(predlog_test); upper_c = np.zeros_like(predlog_test)
        for gl in group_labels:
            mc, mt = (g_cal==gl), (g_test==gl)
            if mc.sum() < 10 or mt.sum()==0: continue
            qgn = conformal_q(norm_scores_cal[mc], a)
            lower_c[mt] = predlog_test[mt] - qgn*(std_test[mt]+eps)
            upper_c[mt] = predlog_test[mt] + qgn*(std_test[mt]+eps)
        lower_c = np.exp(lower_c)-offset; upper_c = np.exp(upper_c)-offset
        cov_combined[a].append(np.mean((y_test>=lower_c)&(y_test<=upper_c))*100)
        width_combined[a].append(np.mean(upper_c-lower_c))

print("time", time.time()-t0, flush=True)
summary = {}
for a in alphas:
    cv, wv = np.array(cov_vanilla[a]), np.array(width_vanilla[a])
    cc, wc = np.array(cov_combined[a]), np.array(width_combined[a])
    key = f"{int((1-a)*100)}"
    summary[key] = {'target': (1-a)*100,
        'vanilla_cov_mean': float(cv.mean()), 'vanilla_cov_std': float(cv.std()), 'vanilla_width_mean': float(wv.mean()),
        'combined_cov_mean': float(cc.mean()), 'combined_cov_std': float(cc.std()), 'combined_width_mean': float(wc.mean())}
    print(f"target {(1-a)*100:.0f}%: vanilla={cv.mean():.2f}+/-{cv.std():.2f} (w={wv.mean():.2f})  combined={cc.mean():.2f}+/-{cc.std():.2f} (w={wc.mean():.2f})", flush=True)

with open('artifacts/step21_reliability_final.json','w') as f:
    json.dump(summary, f, indent=2)
