import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

d = json.load(open('artifacts/bench6_models.json'))
names = ['Random Forest', 'Extra Trees', 'Gradient Boosting (Hist)', 'k-Nearest Neighbors', 'Ridge Regression', 'Support Vector Regression']
short = ['RF', 'Extra\nTrees', 'Grad.\nBoost.', 'k-NN', 'Ridge', 'SVR']
r2 = [d[n]['r2'] for n in names]
mae = [d[n]['mae'] for n in names]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
colors = ['#4C72B0']*6
colors[1] = '#55A868'  # highlight Extra Trees (the winner)

axes[0].bar(short, r2, color=colors)
axes[0].set_ylabel('$R^2$ (held-out)')
axes[0].set_ylim(0.6, 1.0)
axes[0].set_title('(a) Point-prediction accuracy')

axes[1].bar(short, mae, color=colors)
axes[1].set_ylabel('MAE (K, held-out)')
axes[1].set_title('(b) Mean absolute error')

plt.tight_layout()
plt.savefig('paper/figures/fig3_benchmark.pdf')
plt.savefig('paper/figures/fig3_benchmark.png', dpi=200)
plt.close()
print("done")
