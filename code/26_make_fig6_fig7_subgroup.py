import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

d = json.load(open('artifacts/step20_final_eval.json'))
groups = ['A(<=2 elem)','B(3 elem)','C(4 elem)','D(5 elem)','E(6 elem)','F(>=7 elem)']
labels = ['$\\leq$2', '3', '4', '5', '6', '$\\geq$7']

vanilla_cov = [d['vanilla_marginal']['per_group'][g]['cov'] for g in groups]
mondrian_cov = [d['mondrian_unnorm']['per_group'][g]['cov'] for g in groups]
combined_cov = [d['normalized_mondrian']['per_group'][g]['cov'] for g in groups]

vanilla_w = [d['vanilla_marginal']['per_group'][g]['width'] for g in groups]
combined_w = [d['normalized_mondrian']['per_group'][g]['width'] for g in groups]

# Figure 6: per-group coverage, 3 methods
x = np.arange(len(groups))
width = 0.25
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.bar(x-width, vanilla_cov, width, label='Vanilla marginal CP', color='#C44E52')
ax.bar(x, mondrian_cov, width, label='Mondrian CP', color='#55A868')
ax.bar(x+width, combined_cov, width, label='Normalized+Mondrian (proposed)', color='#4C72B0')
ax.axhline(80, linestyle='--', color='gray', linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(60, 90)
ax.set_xlabel('Number of elements in composition')
ax.set_ylabel('Coverage (%)')
ax.set_title('Per-group coverage at 80% target (leak-free evaluation)')
ax.legend(fontsize=7.5, loc='upper right')
plt.tight_layout()
plt.savefig('paper/figures/fig6_group_coverage.pdf')
plt.savefig('paper/figures/fig6_group_coverage.png', dpi=200)
plt.close()

# Figure 7: per-group width, vanilla vs combined
fig, ax = plt.subplots(figsize=(6.0, 4.0))
width2 = 0.32
ax.bar(x-width2/2, vanilla_w, width2, label='Vanilla marginal CP', color='#C44E52')
ax.bar(x+width2/2, combined_w, width2, label='Normalized+Mondrian (proposed)', color='#4C72B0')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel('Number of elements in composition')
ax.set_ylabel('Average interval width (K)')
ax.set_title('Interval width by compositional group at 80% target')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('paper/figures/fig7_group_width.pdf')
plt.savefig('paper/figures/fig7_group_width.png', dpi=200)
plt.close()
print("Fig6/Fig7 done")
