import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

d = json.load(open('artifacts/et_reliability.json'))
targets = [70, 80, 90]
vanilla = [d[str(t)]['vanilla_cov_mean'] for t in targets]
combined = [d[str(t)]['combined_cov_mean'] for t in targets]

fig, ax = plt.subplots(figsize=(4.6, 4.0))
ax.plot(targets, targets, '--', color='gray', linewidth=1, label='Ideal')
ax.plot(targets, vanilla, 'o-', color='#C44E52', label='Vanilla marginal CP')
ax.plot(targets, combined, 's-', color='#4C72B0', label='Normalized + Mondrian (proposed)')
ax.set_xlabel('Target confidence level (%)')
ax.set_ylabel('Achieved marginal coverage (%)')
ax.set_title('Reliability diagram (300 repeated splits)')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(68,92); ax.set_ylim(68,92)
plt.tight_layout()
plt.savefig('paper/figures/fig5_reliability.pdf')
plt.savefig('paper/figures/fig5_reliability.png', dpi=200)
plt.close()
print("done")
