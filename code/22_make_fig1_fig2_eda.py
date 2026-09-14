import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

train = pd.read_csv('/mnt/user-data/uploads/train.csv')

# Figure 1: distribution of critical_temp
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.hist(train['critical_temp'], bins=60, color='#4C72B0', edgecolor='white', linewidth=0.3)
ax.set_xlabel('Critical temperature, $T_c$ (K)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of superconducting critical temperature')
plt.tight_layout()
plt.savefig('paper/figures/fig1_distribution.pdf')
plt.savefig('paper/figures/fig1_distribution.png', dpi=200)
plt.close()

# Figure 2: heteroscedasticity by number_of_elements group
groups = train.groupby('number_of_elements')['critical_temp']
means = groups.mean(); stds = groups.std(); counts = groups.count()
noe = means.index.values

fig, ax1 = plt.subplots(figsize=(5.6, 3.8))
bp_data = [train.loc[train['number_of_elements']==n, 'critical_temp'].values for n in noe]
bp = ax1.boxplot(bp_data, positions=noe, widths=0.6, showfliers=False, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#8FAADC')
    patch.set_alpha(0.8)
ax1.set_xlabel('Number of elements in composition')
ax1.set_ylabel('Critical temperature, $T_c$ (K)')
ax1.set_title('Critical temperature by compositional complexity')
ax1.set_xticks(noe)
plt.tight_layout()
plt.savefig('paper/figures/fig2_group_heteroscedasticity.pdf')
plt.savefig('paper/figures/fig2_group_heteroscedasticity.png', dpi=200)
plt.close()

print("Fig1/Fig2 done")
print(pd.DataFrame({'n_elements':noe,'count':counts.values,'mean_Tc':means.values,'std_Tc':stds.values}).to_string())
