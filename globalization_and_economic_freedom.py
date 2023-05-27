#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/05/27/globalization-and-economic-freedom/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl

# Read all the data
#
# Economic freedom scores are all in one place
ecofree = pd.read_csv('freedom-scores.csv', header=0)

# Globalization scores are also all in one place
glob = pd.read_excel('KOFGI_2022_public.xlsx', header=0)

# OK, select the years for analysis
ecofree_year = 2020
glob_year = 2000

# Select the data
x = ecofree.loc[ecofree['Index Year']==ecofree_year]
x.index = x['Short Name']
# rename index so that it matches with the other dataframe
x.index.name = 'country'
x = x.drop(columns='Short Name')
#
y = glob.loc[glob['year']==glob_year]
y.index = y['country']
y = y.drop(columns='country')

# Merge the data, keep only common countries
merged_data = x.merge(y, how='inner', left_on='country', right_on='country')
print(merged_data.corr()['KOFGI'])

# Select variables to use, fix the axes presentation
set_econname = 'Government Integrity'
set_globname = 'KOFGI'
if ecofree_year > glob_year:
    yname = set_econname
    xname = set_globname
else:
    yname = set_globname
    xname = set_econname
# Use only the required variables
xy = merged_data[[xname, yname, 'ISO Code']].dropna()
# Compute the cross-correlation
corr = xy[[xname, yname]].corr().iloc[1, 0]
# Fix the titles for the plot
if ecofree_year > glob_year:
    set_main_title = yname+' in '+str(ecofree_year)+' vs. Globalization in '+str(glob_year)+', ρ = '+str(round(corr,2))
    set_x_label = 'Globalization, index'
    set_y_label = yname+', index'
else:
    set_main_title = 'Globalization in '+str(glob_year)+' vs. '+xname+' in '+str(ecofree_year)+', ρ = '+str(round(corr,2))
    set_y_label = 'Globalization, index'
    set_x_label = xname+', index'

# Done, now for the plot
f, ax = plt.subplots(figsize=[8, 5])
xx = xy[xname]
yy = xy[yname]
fontsize = 10
plot_handler = ax.scatter(x=xx, y=yy, s=xx*(1.62*5), alpha=1.0, color='green')
ax.set_title(set_main_title, fontsize=fontsize, fontweight='bold')
ax.set_xlabel(set_x_label, fontweight='bold', fontsize=fontsize)
ax.set_ylabel(set_y_label, fontweight='bold', fontsize=fontsize)
ax.set_ylim([yy.min()-10, yy.max()+10])
ax.grid(True, which='both', color='black', linestyle=':')
lbl = xy['ISO Code'].tolist()
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
    xytext=(10, 20),  ha='right', fontweight='bold', fontsize=10)
    ann_list.append(ann)
    l_i += 1
f.tight_layout()
plt.show()

# You can print (optionally save, remove the comment) the data ranked by the x-variable
sorted_xy = xy.sort_values(by=xname)
print(round(sorted_xy, 2))
# sorted_xy.to_csv(xname+' and '+yname+'.csv')