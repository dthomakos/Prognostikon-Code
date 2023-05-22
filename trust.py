#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/05/22/in-government-we-trust/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

# Read the OECD data files
trust = pd.read_csv('DP_LIVE_22052023220521471.csv', header=0)
ineq  = pd.read_csv('DP_LIVE_22052023220651049.csv', header=0)
dinc  = pd.read_csv('DP_LIVE_22052023220348330.csv', header=0)

# Get the latest available data
trust_last = trust.groupby(by='LOCATION').apply(lambda x: x['Value'].iloc[-1])
ineq_last  = ineq.groupby(by='LOCATION').apply(lambda x: x['Value'].iloc[-1])
dinc_last  = dinc.groupby(by='LOCATION').apply(lambda x: x['Value'].iloc[-1]/1e+3)

# Put together, give nice names
all = pd.concat([trust_last, ineq_last, dinc_last], axis=1).dropna()
all.columns = ['Trust', 'Inequality', 'Disp. Income']

# Remove outliers?
remove_outliers = True

# Let's create a scatterplot, for inequality first
if remove_outliers:
    use = all.drop(labels=['TUR', 'MEX', 'CHL', 'CRI', 'SVK', 'SVN', 'CZE'])
else:
    use = all
corr = use.corr().loc['Inequality', 'Trust']
f, ax = plt.subplots(figsize=[20, 12.3])
xx = use['Inequality']
yy = use['Trust']
fontsize = 12
plot_handler = ax.scatter(x=xx, y=yy, s=yy*20, alpha=1.0, color='red')
ax.set_title('Trust in Government vs. Income Inequality, ρ = '+str(round(corr,2)), fontsize=fontsize, fontweight='bold')
ax.set_xlabel('Income Inequality, Gini', fontweight='bold', fontsize=fontsize)
ax.set_ylabel('Trust in Government, %', fontweight='bold', fontsize=fontsize)
ax.set_ylim([20, 90])
ax.grid(True, which='both', color='black', linestyle=':')
lbl = use.index.tolist()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
ax.set_facecolor("lightblue")
# Carefull with the annotation!!!
l_i = 0
ann_list = []
for z1, z2 in zip(xx, yy):
    label = lbl[l_i]
    ann = ax.annotate(label, (z1, z2), textcoords="offset points",
    xytext=(12, 22),  ha='right', fontweight='bold', fontsize=12)
    ann_list.append(ann)
    l_i += 1
f.tight_layout()
plt.show()

# and then for disposable income
if remove_outliers:
    use = all.drop(labels=['MEX', 'CRI', 'USA'])
else:
    use = all
corr = use.corr().loc['Disp. Income', 'Trust']
f, ax = plt.subplots(figsize=[20, 12.3])
xx = use['Disp. Income']
yy = use['Trust']
fontsize = 12
plot_handler = ax.scatter(x=xx, y=yy, s=yy*20, alpha=1.0, color='red')
ax.set_title('Trust in Government vs. Disposable Income, ρ = '+str(round(corr,2)), fontsize=fontsize, fontweight='bold')
ax.set_xlabel('Disposable Income, current USD', fontweight='bold', fontsize=fontsize)
ax.set_ylabel('Trust in Government, %', fontweight='bold', fontsize=fontsize)
ax.set_ylim([20, 90])
ax.grid(True, which='both', color='black', linestyle=':')
lbl = use.index.tolist()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
ax.set_facecolor("lightblue")
# Carefull with the annotation!!!
l_i = 0
ann_list = []
for z1, z2 in zip(xx, yy):
    label = lbl[l_i]
    ann = ax.annotate(label, (z1, z2), textcoords="offset points",
    xytext=(12, 22),  ha='right', fontweight='bold', fontsize=12)
    ann_list.append(ann)
    l_i += 1
f.tight_layout()
plt.show()