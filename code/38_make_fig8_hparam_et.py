import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

imp = json.load(open('artifacts/et_hparam_importance.json'))
names = list(imp.keys()); vals = list(imp.values())
order = sorted(range(len(vals)), key=lambda i: vals[i])
names = [names[i] for i in order]; vals = [vals[i] for i in order]

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.barh(names, vals, color='#55A868')
ax.set_xlabel('Relative importance (fANOVA)')
ax.set_title('Extra Trees hyperparameter importance\n(248 Optuna trials)')
plt.tight_layout()
plt.savefig('paper/figures/fig8_hparam_importance.pdf')
plt.savefig('paper/figures/fig8_hparam_importance.png', dpi=200)
plt.close()
print("done:", dict(zip(names,vals)))
