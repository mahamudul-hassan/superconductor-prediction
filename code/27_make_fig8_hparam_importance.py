import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.load_study(study_name='rf_tuning', storage='sqlite:////home/claude/artifacts/optuna_study.db')
imp = optuna.importance.get_param_importances(study)
names = list(imp.keys()); vals = list(imp.values())

fig, ax = plt.subplots(figsize=(5.2, 3.4))
order = sorted(range(len(vals)), key=lambda i: vals[i])
names = [names[i] for i in order]; vals = [vals[i] for i in order]
ax.barh(names, vals, color='#4C72B0')
ax.set_xlabel('Relative importance (fANOVA)')
ax.set_title(f'Random-forest hyperparameter importance\n({len(study.trials)} Optuna trials, leak-safe validation set)')
plt.tight_layout()
plt.savefig('paper/figures/fig8_hparam_importance.pdf')
plt.savefig('paper/figures/fig8_hparam_importance.png', dpi=200)
plt.close()
print("Fig8 done:", dict(zip(names,vals)))
