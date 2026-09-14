import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

d = np.load('artifacts/et_pool_tuned.npz')
y_pool, pred_raw_pool = d['y_pool'], d['pred_raw_pool']
r2 = r2_score(y_pool, pred_raw_pool)
mae = mean_absolute_error(y_pool, pred_raw_pool)

fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.scatter(y_pool, pred_raw_pool, s=6, alpha=0.35, color='#55A868', linewidths=0)
lims = [0, max(y_pool.max(), pred_raw_pool.max())*1.02]
ax.plot(lims, lims, '--', color='crimson', linewidth=1.2)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Actual $T_c$ (K)')
ax.set_ylabel('Predicted $T_c$ (K)')
ax.set_title(f'Parity plot, held-out set (Extra Trees)\n$R^2$={r2:.3f}, MAE={mae:.2f} K')
plt.tight_layout()
plt.savefig('paper/figures/fig4_parity.pdf')
plt.savefig('paper/figures/fig4_parity.png', dpi=200)
plt.close()
print("Fig4 done. R2=", r2, "MAE=", mae)
