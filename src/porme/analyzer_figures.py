# %% defs
# -*- coding: utf-8 -*-
# Code best viewed at 100 characters per line
"""
Created on Thu Jul  7 14:24:12 2022

@author: MPardo
"""

import matplotlib.lines as lines
import matplotlib.gridspec as grid_spec
import os
import cv2
import numpy as np
import pandas as pd
import funcs as f
import warnings
from funcs import show, sho2
import matplotlib.pyplot as plt

SOLV = ["ACE", "DCM", "MINI", "PCL", ]
CONC = [0, 10, 20]
TEMP = [90, 110, 130]

_errors = {"EMPTY": "Empty", "CIRC": "Circularity", "SMALL": "Small", "SPLIT": "Split",
           "missing img": "Other", "cnt fail": "Other", "th fail": "Other"}
_names = ['Other', 'Circularity', 'Small', 'Split', 'Empty', ]

def valid_combo(_mat, _conc, _temp):
    if _mat == "PCL" or _mat == "MINI":
        if _conc > 0:
            return False
    if _conc > 0 and _temp < 110:
        return False
    else:
        return True


def colorline(cmap: str, fac: float):
    if len(cmap) == 7:
        r = cmap[1:3]
        g = cmap[3:5]
        b = cmap[5:7]
    else:
        r = cmap[1]
        g = cmap[2]
        b = cmap[3]

    out = "#"
    for color in ("r", "g", "b"):
        _temp = locals()[color]
        _temp = int(_temp, 16)
        _temp = int(_temp * fac)
        _temp = hex(_temp)[2:]
        _temp = _temp + "0" if _temp == "0" else _temp
        out += _temp

    return out


def colormap(cmat, cconc, ctemp, alt_ret=None):
    if cmat.lower() == "pcl":
        if str(ctemp) == "90":
            return "#f1d9ff"
        if str(ctemp) == "110":
            return "#cc8aeb"
        if str(ctemp) == "130":
            return "#903eb8"

    if cmat.lower() == "mini":
        if str(ctemp) == "90":
            return "#bbb"
        if str(ctemp) == "110":
            return "#777"
        if str(ctemp) == "130":
            return "#000"

    if cmat.lower() == "ace":
        if str(ctemp) == "90":
            if str(cconc) == "0":
                return "#fff1ab"
            if str(cconc) == "10":
                return "#ffcc00"
            if str(cconc) == "20":
                return "#734800"
        if str(ctemp) == "110":
            if str(cconc) == "0":
                return "#f2bf85"
            if str(cconc) == "10":
                return "#d17034"
            if str(cconc) == "20":
                return "#732600"
        if str(ctemp) == "130":
            if str(cconc) == "0":
                return "#fcb6bf"
            if str(cconc) == "10":
                return "#f2497e"
            if str(cconc) == "20":
                return "#b8002b"

    if cmat.lower() == "dcm":
        if str(ctemp) == "90":
            if str(cconc) == "0":
                return "#bbffff"
            if str(cconc) == "10":
                return "#66cc00"
            if str(cconc) == "20":
                return "#397318"
        if str(ctemp) == "110":
            if str(cconc) == "0":
                return "#8cddff"
            if str(cconc) == "10":
                return "#2491b3"
            if str(cconc) == "20":
                return "#005473"
        if str(ctemp) == "130":
            if str(cconc) == "0":
                return "#abf5c4"
            if str(cconc) == "10":
                return "#45e693"
            if str(cconc) == "20":
                return "#008c4d"

    return alt_ret


# %% Main
user_id = "cojo"

csv_path = r"C:/Users/"+user_id+"/OneDrive - Universidad Adolfo Ibanez/2022/Jose_2/Data/New"
csv_files = [file for file in os.listdir(csv_path) if ".csv" in file]

#%%
data = pd.concat((pd.read_csv(os.path.join(csv_path, file)) for file in csv_files), ignore_index=True)

data = data.loc[:, "Material":"Conversion factor"]  # Needed if index col is present
data["Temperature"] = data["Temperature"].astype(int)
data["Layer"] = data["Layer"] + 1
data = data[(data["Layer"] != 3)]

data = data.set_index(
    [
        "Material",
        "Temperature",
        "%BG",
        "Sample",
        "Location",
        "Layer",
        "Column",
        "Row"
    ])
# Use as base to group
data.loc[data["Print direction"] == "Vertical",  "Print direction"] = 0
data.loc[data["Print direction"] == "Horizontal", "Print direction"] = 1
data["Print direction"] = data["Print direction"].astype(float)
data.rename(columns={"Print direction": "Vertical print"}, inplace=True)

data["Valid"] = False
data.loc[data["Pore validity"] == "ok", "Valid"] = True
data["Invalid"] = ~data["Valid"]

data = data.reset_index()

data.to_csv('../DataGraph/DataPoro.csv', index = False)

#%%

data = pd.read_csv('../DataGraph/DataPoro.csv')

# %% Pore size vs layer for all combos
# for _temp in TEMP:
# _aux = data[data["Temperature"] == _temp]
# _aux = _aux.groupby("Layer")
_aux = data.groupby("Layer")
data_avg = _aux.mean().reset_index()
data_std = _aux.std().reset_index()
data_count = _aux.count().reset_index()
data_avg["Vertical print"] = data_avg["Vertical print"].round().astype(int)
data_std["Vertical print"] = data_avg["Vertical print"]
data_count["Vertical print"] = data_avg["Vertical print"]

fig, ax = plt.subplots(dpi=200)
ax.set_ylim(0, 0.5)
ax.set_xlim(2, 57)
_var = "Pore size"
for direction in (0, 1):
    _marker = "o" if direction else "^"
    _label = "Vertical" if direction else "Horizontal"

    _avg = data_avg[data_avg["Vertical print"] == direction].reset_index()
    _std = data_std[data_std["Vertical print"] == direction].reset_index()
    _count = data_count[data_count["Vertical print"] == direction].reset_index()

    # Mean
    _color = "orange" if direction else "royalblue"
    #ax.scatter(_avg["Layer"], _avg[_var], s=50, linewidths=0.25, zorder=100,
               #marker=_marker, color=_color, label=_label, edgecolors="black")
    
    # STD
    _color = "#e90" if direction else "#16e"
    ax.fill_between(_avg["Layer"], _avg[_var] - 1.96*_std[_var]/np.sqrt(_count[_var]),
                    _avg[_var] + 1.96*_std[_var]/np.sqrt(_count[_var]),label=_label,
                color=_color
                   )

    """
    # STD fill between
    _color = "#e90" if direction else "#16e"
    for _sign in (1.0, -1.0):
        ax.fill_between(_avg["Layer"], _avg[_var] - _std[_var]*_sign,  _avg[_var] + _std[_var]*_sign,
                   zorder=100, color=_color, linewidths=0.75,
                   edgecolors=None, alpha=0.2
                   )
        
    # STD
    _color = "#e90" if direction else "#16e"
    for _sign in (1.0, -1.0):
        ax.scatter(_avg["Layer"], _avg[_var] + _std[_var]*_sign,
                   zorder=100, color=_color, linewidths=0.75,
                   edgecolors=None, s=30, marker="_",
                   )
        
    
    # 10th and 90th percentiles
    _color = "#e90" if direction else "#16e"
    for _q in (0.1, 0.9):
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            _aux2 = _aux.quantile(_q).reset_index()
        _aux2 = _aux2[_aux2["Vertical print"] == direction]
        ax.scatter(_aux2["Layer"], _aux2[_var], s=10, zorder=100,
                   marker="o", color=_color, facecolors="none", linewidths=0.75)
    

    # Density
    _data = data[data["Vertical print"] == direction].reset_index()
    _data[_var] = _data[_var] + np.random.uniform(-0.005, 0.005, len(_data))
    _data["Layer"] = _data["Layer"].astype(float) + np.random.uniform(-0.5, 0.5, len(_data))
    _color = "#fc5" if direction else "#5af"
    _data = _data.iloc[::10]
    ax.scatter(_data["Layer"], _data[_var], alpha=1/20, s=4, zorder=10,
               marker=_marker, color=_color, edgecolors=None)
    """

ax.plot([0, 60], [0.4, 0.4], color="red", zorder=10, linewidth=2, linestyle=":",alpha=0.8)
# ax.set_title(f"Pore size vs layer - {_temp}C")
ax.set_title("Pore size vs layer")
ax.set_xlabel("Layer")
ax.set_ylabel("Pore size [mm]")
ax.grid(alpha=0.5, zorder=50)
plt.legend(loc="lower right")
plt.show()

# %% Pore size vs layer for all combos
# for _temp in TEMP:
# _aux = data[data["Temperature"] == _temp]
# _aux = _aux.groupby("Layer")
_aux = data.groupby("Layer")
data_avg = _aux.mean().reset_index()
data_std = _aux.std().reset_index()
data_count = _aux.count().reset_index()
data_avg["Vertical print"] = data_avg["Vertical print"].round().astype(int)
data_std["Vertical print"] = data_avg["Vertical print"]
data_count["Vertical print"] = data_avg["Vertical print"]

fig, ax = plt.subplots(dpi=200)
ax.set_ylim(0, 0.5)
ax.set_xlim(2, 57)
_var = "Pore size"

direction = 0
_marker = "o" if direction else "^"
_label = "Average"

_avg = data_avg.reset_index()
_std = data_std.reset_index()
_count = data_count.reset_index()

#Smooth of curve
kernel_size = 2
kernel = np.ones(kernel_size) / kernel_size
_avg_fix = np.convolve(_avg[_var], kernel, mode='same')

_avg_fix[0] = _avg_fix[0]*2

# Mean
_color = "orange" if direction else "royalblue"
#ax.scatter(_avg["Layer"], _avg[_var], s=50, linewidths=0.25, zorder=100,
           #marker=_marker, color=_color, label=_label, edgecolors="black")

# STD
_color = "#e90" if direction else "#16e"
ax.plot(_avg["Layer"], _avg_fix, color="Black",label="Mean")
ax.fill_between(_avg["Layer"], _avg_fix - _std[_var],
                _avg_fix + _std[_var],label="SD",
            color="black",alpha=0.3
               )

"""
# STD fill between
_color = "#e90" if direction else "#16e"
for _sign in (1.0, -1.0):
    ax.fill_between(_avg["Layer"], _avg[_var] - _std[_var]*_sign,  _avg[_var] + _std[_var]*_sign,
               zorder=100, color=_color, linewidths=0.75,
               edgecolors=None, alpha=0.2
               )
    
# STD
_color = "#e90" if direction else "#16e"
for _sign in (1.0, -1.0):
    ax.scatter(_avg["Layer"], _avg[_var] + _std[_var]*_sign,
               zorder=100, color=_color, linewidths=0.75,
               edgecolors=None, s=30, marker="_",
               )
    

# 10th and 90th percentiles
_color = "#e90" if direction else "#16e"
for _q in (0.1, 0.9):
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore")
        _aux2 = _aux.quantile(_q).reset_index()
    _aux2 = _aux2[_aux2["Vertical print"] == direction]
    ax.scatter(_aux2["Layer"], _aux2[_var], s=10, zorder=100,
               marker="o", color=_color, facecolors="none", linewidths=0.75)


# Density
_data = data[data["Vertical print"] == direction].reset_index()
_data[_var] = _data[_var] + np.random.uniform(-0.005, 0.005, len(_data))
_data["Layer"] = _data["Layer"].astype(float) + np.random.uniform(-0.5, 0.5, len(_data))
_color = "#fc5" if direction else "#5af"
_data = _data.iloc[::10]
ax.scatter(_data["Layer"], _data[_var], alpha=1/20, s=4, zorder=10,
           marker=_marker, color=_color, edgecolors=None)
"""

ax.plot([0, 60], [0.4, 0.4], color="red", zorder=10, linewidth=2, linestyle=":",alpha=0.8)
# ax.set_title(f"Pore size vs layer - {_temp}C")
ax.set_title("Pore size vs layer")
ax.set_xlabel("Number of Layer")
ax.set_ylabel("Pore size [mm]")
ax.grid(alpha=0.5, zorder=50)
plt.legend(loc="lower right")
plt.show()

#%%

print(1000*data["Pore size"].mean())
print(1000*data["Pore size"].std()*1.96/np.sqrt(data["Pore size"].count()))

# %% Pore validity vs layer for all combos
total_by_layer = data.groupby("Layer").sum()
total_by_layer = total_by_layer["Valid"] + total_by_layer["Invalid"]

_data = data[data["Pore validity"] != "ok"]
_data.loc[(_data["Pore validity"] == "missing img") |
          (_data["Pore validity"] == "cnt fail") |
          (_data["Pore validity"] == "th fail") , "Pore validity"] = "OTHER"
_data = _data.groupby(["Layer", "Pore validity"]).sum()["Invalid"]
_data = _data / total_by_layer * 100
_data = _data.reset_index()
_data.rename(columns={0: "Invalid"}, inplace=True)

fig, ax = plt.subplots(dpi=200)
ax.set_xlim(2, 57)

_color = {
    "SPLIT":"C4",
    "SMALL": "C2",
    "EMPTY":"C3",
    "CIRC":"C1",
    "OTHER":"C0",
    }
for _layer in range(2, 56):
    _rolling = 0
    for _type in [
            "SPLIT",
            "SMALL",
            "EMPTY",
            "CIRC",
            "OTHER",
            ]:
        _label = _type.capitalize() if _layer == 55 else None
        _aux = _data[(_data["Layer"] == _layer) & (_data["Pore validity"] == _type)]

        ax.bar(_layer, _aux["Invalid"], bottom=_rolling, label=_label,
                color=_color[_type]
               )
        _rolling += float(_aux["Invalid"]) if len(_aux) != 0 else 0

ax.set_title("Invalid pores by layer")
ax.set_xlabel("Layer")
ax.set_ylabel("Invalid pores [%]")
ax.grid(alpha=0.5, zorder=50, linestyle="--", linewidth=0.5)
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
plt.show()

# %% Pore size frequency prep
data2 = data.reset_index().set_index(["Material", "Temperature", "%BG", ])
data2["Pore size"] = data2["Pore size"].round(3)
data2 = data2.sort_values([
    "Material",
    "%BG",
    "Temperature",
])

pore_hists = pd.DataFrame()
for combo in data2.index.unique():
    # print(combo)
    _mat, _temp, _conc = combo
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore")
        data_plot = data2.loc[combo, "Pore size"].dropna()

    _lims = (0.2, 0.6)
    _hist = list(np.histogram(data_plot, bins=int((_lims[1] - _lims[0]) * 1001), range=_lims))
    for __i in range(20):
        _hist[0] = cv2.blur(_hist[0], f.k2(7)).ravel()

    _aux = np.array([_mat, _temp, _conc])
    _aux = np.broadcast_to(_aux, (400, 3))
    _aux = np.hstack((_aux, (_hist[1][:-1][:, None] * 1000).astype(int), _hist[0][:, None]))
    _aux = pd.DataFrame(_aux, columns=["Material", "Temperature", "%BG", "Pore size", "Frequency"])
    pore_hists = pd.concat([pore_hists, _aux], ignore_index=True)

for _col in pore_hists.columns[1:]:
    pore_hists[_col] = pore_hists[_col].astype(int)

# %% Pore size frequency plot
gs = grid_spec.GridSpec(len(data2.index.unique()), 1)
fig = plt.figure(figsize=(6, 6), dpi=175)
paper_info = pd.DataFrame()

axes = []
for _num, (_mat, _temp, _conc) in enumerate(data2.index.unique()):
    _aux = pore_hists[
        (pore_hists["Material"] == _mat) &
        (pore_hists["Temperature"] == int(_temp)) &
        (pore_hists["%BG"] == int(_conc))
    ]

    # print(_mat, _temp, _conc, _aux["Frequency"].sum())
    # creating new axes object
    axes.append(fig.add_subplot(gs[_num:_num+1, 0:]))

    # plotting the distribution
    axes[-1].plot(_aux["Pore size"],
                  _aux["Frequency"]/_aux["Frequency"].sum(),
                  color=colorline(colormap(_mat, _conc, _temp), 0.7),
                  lw=1
                  )
    axes[-1].fill_between(_aux["Pore size"],
                          _aux["Frequency"]/_aux["Frequency"].sum(),
                          alpha=0.85,
                          color=colormap(_mat, _conc, _temp)
                          )
    
    a = round(_aux[_aux["Frequency"] == _aux["Frequency"].max()]["Pore size"].mean(),0)
    b = _aux[(_aux["Pore size" ] > a-5) & (_aux["Pore size" ] < a+5)]
    
    paper_info_aux = pd.DataFrame(data={"Material":_aux["Material"],
                                        "Temperature":_aux["Temperature"],
                                        "%BG":_aux["%BG"],
                                        "Pore":b["Pore size"].mean(),
                                        "Freq":b["Frequency"].sum()
                                        })
    
    paper_info= paper_info.append(paper_info_aux, ignore_index=True)

    # setting uniform x and y lims
    axes[-1].set_xlim(200, 600)
    axes[-1].set_ylim(0, 0.0175)

    # make background transparent
    rect = axes[-1].patch
    rect.set_alpha(0)

    # remove borders, axis ticks, and labels
    axes[-1].set_yticklabels([])
    axes[-1].set_yticks([])

    if _num == len(data2.index.unique())-1:
        axes[-1].set_xlabel("Pore size [µm]", fontsize=12)
        # axes[-1].set_xticklabels([data["Pore size"].mean()*1000])
        # axes[-1].set_xticks([data["Pore size"].mean()*1000])
    else:
        axes[-1].set_xticklabels([])
        axes[-1].set_xticks([])

    spines = ["top", "right", "left", "bottom"]
    for s in spines:
        axes[-1].spines[s].set_visible(False)

    _space1 = "  " if _conc == 0 else ""
    _space2 = " " if _mat in ("PCL", "ACE") else ""
    _space3 = "  " if _temp == 90 else ""
    axes[-1].text(100, 0, f"{_mat}{_space2} {_space1}{_conc}% {_space3}{_temp}°C", fontsize=10)

gs.update(hspace=-0.7)
fig.add_artist(lines.Line2D([0.512, 0.512], [0.12, 0.85], color="red", alpha=0.4, linestyle="--"))
fig.add_artist(lines.Line2D([0.4425, 0.4425], [0.12, 0.85],
               color="black", alpha=0.4, linestyle="--"))
fig.suptitle("Normalized distributions of pore size", x=0.4, y=0.925, fontsize=14)
fig.supylabel("Material", x=-0.13, y=0.46)
plt.show()

# %% Pore size frequency plot average

fig = plt.figure(figsize=(5, 4), dpi=300)
graph_data = pore_hists.groupby(["Pore size"]).sum().reset_index()

plt.plot(graph_data["Pore size"],graph_data["Frequency"])

max_value = graph_data.sort_values("Frequency", ascending=False)
plt.plot([max_value.iloc[0,0],max_value.iloc[0,0]],[0,3700], "--", color="Red")

plt.ylim(0,3650)
plt.xlabel("Pore size [µm]")
plt.ylabel("Frequency")

#%%
color_temp = {
    90:"C1",
    110:"C2",
    130:"C3",
    }

fig = plt.figure(figsize=(5, 4), dpi=300)
for _temp in [90,110,130]:
    _aux = pore_hists[
        (pore_hists["Temperature"] == int(_temp)) &
        ((pore_hists["%BG"] == int(0)))
    ]
    
    _aux = _aux.groupby(["Pore size"]).sum().reset_index()
    
    #print(_mat, _temp, _conc, _aux["Frequency"].sum())
    # creating new axes object

    # plotting the distribution
    plt.plot(_aux["Pore size"],
                  _aux["Frequency"],
                  color=color_temp[_temp],
                  label=str(_temp) +" °C"
                  )
    
    print(_aux["Frequency"].sum())
    graph_data = graph_data.groupby(["Pore size"]).sum().reset_index()
    
    max_value = _aux.sort_values("Frequency", ascending=False)
    print(max_value.iloc[0,0])
    #plt.plot([max_value.iloc[0,0],max_value.iloc[0,0]],[0,0.01], "--", color="Red")
    plt.plot([400,400],[0,720], "--", color="blacK")

plt.legend(title="Temperature")
#plt.ylim(0,2000)
plt.xlabel("Pore size [µm]")
plt.ylabel("Frequency")
#%%
data_graph = data2.reset_index()[data2.reset_index()["%BG"] == 0]
print(data_graph.groupby(["Temperature"])["Pore size"].mean().reset_index())
print(data_graph.groupby(["Temperature"])["Pore size"].std().reset_index())

import seaborn as sns
data_graph = data_graph.rename(columns={"Pore size":"Pore"})

sns.boxplot(data_graph, x="Temperature",y="Pore",notch=True)

import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('Pore ~ C(Temperature)', data=data_graph).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

from bioinfokit.analys import stat
# perform multiple pairwise comparison (Tukey's HSD)
# unequal sample size data, tukey_hsd uses Tukey-Kramer test
res = stat()
res.tukey_hsd(df=data_graph, res_var='Pore', xfac_var='Temperature', anova_model='Pore ~ C(Temperature)')
print(res.tukey_summary)

#%% Manage data from R

PATH2 = "C:/Users/"+user_id+"/OneDrive - Universidad Adolfo Ibanez/2022/Jose_2/Data/R"

a = [1,2,3]
b = ["YStress","YStrain","EMod"]

data_new = pd.DataFrame()

for b_i in b:   
    data_r = pd.read_csv(PATH2+"/"+b_i+"_pwc1_emmean.csv")
    
    data_r["Temperature"] = data_r["group"].str.split("_",expand=True)[0].astype(int)
    #data_r["Solvent"] = data_r["group"].str.split("_",expand=True)[0]
    data_r["Concentration"] = data_r["group"].str.split("_",expand=True)[1].astype(int)
    
    data_r["var"] = b_i
    
    data_new = data_new.append(data_r, ignore_index=True)

#data_new = data_new.rename(columns=({"Temp":"Temperature"}))
data_new = data_new.rename(columns=({"Mat":"Solvent"}))

data_new["conf"] = data_new["conf.high"]-data_new["conf.low"]

#%% Manage data from R for 0BG

PATH2 = "C:/Users/"+user_id+"/OneDrive - Universidad Adolfo Ibanez/2022/Jose_2/Data/R"

a = [1,2,3]
b = ["YStress","YStrain","EMod"]

data_new_0 = pd.DataFrame()

for b_i in b:   
    data_r = pd.read_csv(PATH2+"/"+b_i+"_pwc1_emmean_0.csv")
    
    data_r["Temperature"] = data_r["group"]
    #data_r["Solvent"] = data_r["group"].str.split("_",expand=True)[0]
    
    data_r["var"] = b_i
    
    data_new_0 = data_new_0.append(data_r, ignore_index=True)

#data_new = data_new.rename(columns=({"Temp":"Temperature"}))
data_new_0 = data_new_0.rename(columns=({"Mat":"Solvent"}))

data_new_0["conf"] = data_new_0["conf.high"]-data_new_0["conf.low"]


#%% Fix data for tables
PATH2 = "C:/Users/"+user_id+"/OneDrive - Universidad Adolfo Ibanez/2022/Jose_2/Data/R"

groups = ["Solvent","Temperature","Concentration"]

#For no 0 BG
#data_new_0 = data_new

data_pore = data2.reset_index().rename(columns={"Material":"Solvent","%BG":"Concentration"})
data_pore = data_pore[["Solvent","Temperature","Concentration","Pore size"]]
data_pore["Pore size"] = data_pore["Pore size"]*1000

#For no 0 BG
#data_pore = data_pore[(data_pore["Solvent"] == "ACE") | (data_pore["Solvent"] == "DCM")] 
#data_pore = data_pore[(data_pore["Temperature"] == 110) | (data_pore["Temperature"] == 130)] 

#Edit for Pool
data_pore_avg = data_pore.groupby(groups).mean().reset_index()
data_pore_std = data_pore.groupby(groups).std().reset_index()

data_all = pd.read_csv("C:/Users/"+user_id+"/OneDrive - Universidad Adolfo Ibanez/2022/Jose_2/Data/Pardo/Data_all.csv")

#For no 0 BG
#data_all = data_all[(data_all["Solvent"] == "ACE") | (data_all["Solvent"] == "DCM")] 
#data_all = data_all[(data_all["Temperature"] == 110) | (data_all["Temperature"] == 130)] 

data_all["YStrain"] = 100*data_all["YStrain"]
data_all["Porosity"] = 100*data_all["Porosity"]

#Edit for pool
data_all_avg = data_all.groupby(groups).mean().reset_index()
data_all_std = data_all.groupby(groups).std().reset_index()

columns = ["Solvent","Temperature","Concentration","Porosity","YStrain"]

data_all_avg = data_all_avg[columns].sort_values(by=groups)
data_all_std = data_all_std[columns].sort_values(by=groups)

#For 0 BG
data_new_0["Concentration"] = 0
data_new_0["se"] = data_new_0["se"]*np.sqrt(6)

for var in ["EMod","YStress"]:
    data_filt = data_new_0[data_new_0["var"] == var]
    data_filt = data_filt[["Solvent","Temperature","Concentration","se","emmean"]]
    data_filt_avg = data_filt[["Solvent","Temperature","Concentration","emmean"]].rename(columns={"emmean":var})
    data_filt_std = data_filt[["Solvent","Temperature","Concentration","se"]].rename(columns={"se":var})
    
    #data_filt_avg = data_filt_avg.groupby(groups).mean().reset_index()
    
    #data_filt_std = ((data_filt_std.groupby(groups).sum()**2/6)**(1/2)).reset_index()
    
    data_all_avg = pd.merge(data_all_avg, data_filt_avg, on =groups)
    data_all_std = pd.merge(data_all_std, data_filt_std, on =groups)

data_pore_avg = pd.merge(data_all_avg, data_pore_avg, on =groups)
data_pore_std = pd.merge(data_all_std, data_pore_std, on =groups)


#%% save CSV

data_pore_avg.to_csv("../DataGraph/data_pore_avg0.csv", index = False)
data_pore_std.to_csv("../DataGraph/data_pore_std0.csv", index = False)

#%%
var = ['Concentration']
data_all = data_all[(data_all["Solvent"] == "ACE") | (data_all["Solvent"] == "DCM")] 
data_pore = data_pore[(data_pore["Solvent"] == "ACE") | (data_pore["Solvent"] == "DCM")] 

data_all = data_all[(data_all["Temperature"] == 110) | (data_all["Temperature"] == 130)] 
data_pore = data_pore[(data_pore["Temperature"] == 110) | (data_pore["Temperature"] == 130)] 

data_all_pool_avg = data_all.groupby(var).mean().reset_index()
data_all_pool_std = data_all.groupby(var).std().reset_index()

data_pool_pore_avg = data_pore.groupby(var).mean().reset_index()
data_pool_pore_std = data_pore.groupby(var).std().reset_index()


data_all_pool_avg = pd.merge(data_all_pool_avg, data_pool_pore_avg, on =var)
data_all_pool_std = pd.merge(data_all_pool_std, data_pool_pore_std, on =var)

#%% PLot to see if good

import docx

# open an existing document
doc = docx.Document()

data_table = data_pore_avg[["Solvent","Temperature","Concentration","Porosity","Pore size","EMod","YStress","YStrain"]]
data_table_std = data_pore_std[["Solvent","Temperature","Concentration","Porosity","Pore size","EMod","YStress","YStrain"]]

#data_table = data_all_pool_avg[["Solvent","Porosity","Pore size","Emod","YStress","YStrain"]]
#data_table_std = data_all_pool_std[["Solvent","Porosity","Pore size","Emod","YStress","YStrain"]]

# add a table to the end and create a reference variable
# extra row is so we can add the header row
t = doc.add_table(data_table.shape[0]+1, data_table.shape[1])

# add the header rows.
for j in range(data_table.shape[-1]):
    t.cell(0,j).text = data_table.columns[j]

# add the rest of the data frame
for i in range(data_table.shape[0]):
    for j in range(data_table.shape[-1]):
        if j > 2:
            t.cell(i+1,j).text = str(round(data_table.values[i,j],1)) + u" \u00B1 " + str(round(data_table_std.values[i,j],1))
        else:
            t.cell(i+1,j).text = str(data_table.values[i,j])

# save the doc
doc.save('C:/Users/"+user_id+"/OneDrive - Universidad Adolfo Ibanez/2022/Jose_2/Data/Pardo/Table4-5.docx')

# %% Grid plot

data_grid = data.groupby(["Column", "Row"])
nan_count = data_grid["Invalid"].sum()
data_grid_std = data_grid.std().reset_index()
data_grid = data_grid.mean()
data_grid["Invalid count"] = round(nan_count / (len(data)/64) * 100, 1)
data_grid = data_grid.reset_index()

black = 30
color = 60
dims = 9 * black + 8 * color
grid = np.zeros((dims, dims, 3), np.uint8)

var = "Pore size"
for _row in range(8):
    for _col in range(8):
        _data = data_grid[(data_grid["Row"] == _row) & (data_grid["Column"] == _col)]

        i_col = slice((_col + 1) * black + (_col + 0) * color,
                      (_col + 1) * black + (_col + 1) * color)
        i_row = slice((_row + 1) * black + (_row + 0) * color,
                      (_row + 1) * black + (_row + 1) * color)

        _val = np.array(np.interp(_data[var],
                                  (data_grid[var].min(), data_grid[var].max()),
                                  (0, 255)), np.uint8)
        grid[i_col, i_row] = cv2.applyColorMap(_val, cv2.COLORMAP_PLASMA)

        _val = str(round(float(_data[var]), 3))
        _val += "%" if var == "Invalid count" else ""

        _common_args = {
            "img": grid, "text": _val,
            "org": ((_row + 1) * black + _row * color + 5,
                    (_col + 1) * black + _col * color + 40),
            "fontFace": cv2.FONT_HERSHEY_PLAIN,
            "fontScale": 1.1,
            "lineType": cv2.LINE_AA, }

        cv2.putText(**_common_args, thickness=2, color=(0, 0, 0))
        cv2.putText(**_common_args, thickness=1, color=(255, 255, 255))
show(grid)

# %% Invalid pores
_aux = data[data["Pore validity"] != "ok"]["Pore validity"]

_values = pd.DataFrame([[0] * len(_names)], columns=_names)

for _key, _val in _errors.items():
    _values[_val] = sum(data["Pore validity"] == _key)

fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
ax.barh(_values.columns, _values.to_numpy().ravel()/len(data)*100, zorder=100)

# Remove axes splines
for s in ['top', 'left', 'right']:
    ax.spines[s].set_visible(False)

# Remove x, y Ticks
ax.yaxis.set_ticks_position('none')

# Add x gridlines
ax.grid(axis="x", color='grey', linestyle='-.', linewidth=0.5, alpha=0.7)

ax.set_title("Invalid pores by category")
ax.set_xlabel("Invalid pores [%]")

plt.show()

#%% Join the data frames

data3 = pd.read_csv("../Pardo/Data_all.csv")
data3 = data3.rename(columns={'Solvent':'Material',
                              'Concentration':'%BG',
                              'Sample':'Location',
                              'Num':'Sample'})

data4 = data.groupby(['Material', 'Temperature', '%BG', 'Sample','Location']).mean()

data5 = pd.merge(data3, data4, on =['Material', 'Temperature', '%BG', 'Sample','Location'])

#%% Plot the info and see correlation between variables
import seaborn as sns
#plt.scatter(data5["Porosity"],data5["Pore size"])
#plt.show()

data6 = data5[["Area","Pore distance","Perimeter","Pore size","Porosity","Height","Diameter","Volume"]]

data6["Porosity"] = data6["Porosity"]*100

#sns.pairplot(data6)
sns.heatmap(data6.corr(), vmin=-1, vmax=1, annot=True)
plt.show()

data7 = data6[["Pore size","Height","Diameter","Porosity"]]
sns.heatmap(data7.corr(), vmin=-1, vmax=1, annot=True)
plt.show()

#%%

sns.pairplot(data7)

#%% 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(data7.iloc[:,:3], data7.iloc[:,-1:], test_size=0.3, random_state=10)

scaler = StandardScaler().fit(X_train) 

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

#%%
reg = linear_model.Lasso(alpha=0.001)
reg.fit(X_train,y_train)
print(reg.coef_)
print(reg.intercept_)

print('R squared training set', round(reg.score(X_train, y_train)*100, 2))
print('R squared test set', round(reg.score(X_test, y_test)*100, 2))

# Training data
pred_train = reg.predict(X_train)
mse_train = mean_squared_error(y_train, pred_train)
print('MSE training set', round(mse_train, 2))

# Test data
pred = reg.predict(X_test)
mse_test =mean_squared_error(y_test, pred)
print('MSE test set', round(mse_test, 2))

#%%

import numpy as np
import matplotlib.pyplot as plt

alphas = np.linspace(0.000001,5000,100)
lasso = linear_model.Lasso(max_iter=10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('Standardized Coefficients')
plt.title('Lasso coefficients as a function of alpha');

#%%

from sklearn.linear_model import LassoCV

# Lasso with 5 fold cross-validation
model = LassoCV(cv=5, random_state=0, max_iter=10000)

# Fit model
model.fit(X_train,y_train.values.ravel())
print(model.alpha_)

#%%
reg = linear_model.Lasso(alpha=model.alpha_)
reg.fit(X_train,y_train)
print(reg.coef_)
print(reg.intercept_)

print('R squared training set', round(reg.score(X_train, y_train)*100, 2))
print('R squared test set', round(reg.score(X_test, y_test)*100, 2))

print(mean_squared_error(y_test, reg.predict(X_test)))

#%%

plt.semilogx(model.alphas_, model.mse_path_, ":")
plt.plot(
    model.alphas_ ,
    model.mse_path_.mean(axis=-1),
    "k",
    label="Average across the folds",
    linewidth=2,
)
plt.axvline(
    model.alpha_, linestyle="--", color="k", label="alpha: CV estimate"
)

plt.legend()
plt.xlabel("alphas")
plt.ylabel("Mean square error")
plt.title("Mean square error on each fold")
plt.axis("tight")

# https://www.kirenz.com/post/2019-08-12-python-lasso-regression-auto/

#%% Final
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(data7.iloc[:,:1], data7.iloc[:,-1:], test_size=0.3, random_state=10)

reg = LinearRegression()
reg.fit(X_train,y_train)
print(reg.coef_)
print(reg.intercept_)

print('R squared training set', round(reg.score(X_train, y_train)*100, 2))
print('R squared test set', round(reg.score(X_test, y_test)*100, 2))

# Training data
pred_train = reg.predict(X_train)
mse_train = mean_squared_error(y_train, pred_train)
print('MSE training set', round(mse_train, 2))

# Test data
pred = reg.predict(X_test)
mse_test =mean_squared_error(y_test, pred)
print('MSE test set', round(mse_test, 2))

#%%

plt.scatter(X_train,y_train)
plt.scatter(X_test,y_test)
plt.show()

#%%

fig = plt.figure(figsize=(5, 4), dpi=300)
scores = (r"$R^2={:.2f}$").format(
            round(reg.score(X_test, y_test), 2),
        )
plt.scatter(data7.iloc[:,-1:],data7.iloc[:,:1]*reg.coef_+reg.intercept_)
plt.plot([35, 70], [35, 70],color="Red")

plt.annotate(scores, xy=(40,180), xycoords='axes points',
            size=14,
            bbox=dict(boxstyle='round', fc='w'))

plt.xlabel("Measured porosity [%]")
plt.ylabel("Predicted porosity [%]")
plt.title("Predicted vs experimental porosity")
plt.show()

#%%
from sklearn.metrics import r2_score

fig = plt.figure(figsize=(5, 4), dpi=300)
scores = (r"$R^2={:.2f}$").format(
            round(r2_score(data7.iloc[:,-1:], data7.iloc[:,:1]*reg.coef_+reg.intercept_), 2),
        )

plt.scatter(data7.iloc[:,-1:],100-245.44*(0.8-data7["Pore size"])**2,alpha=0.5,color="C1",label="Explicit")
plt.scatter(data7.iloc[:,-1:],data7.iloc[:,:1]*reg.coef_+reg.intercept_,label="Machine Learning")
plt.plot([35, 70], [35, 70],color="Red")

plt.annotate(scores, xy=(40,180), xycoords='axes points',
            size=14,
            bbox=dict(boxstyle='round', fc='w'))

plt.legend(loc="lower right")
plt.xlabel("Measured porosity [%]")
plt.ylabel("Predicted porosity [%]")
plt.title("Predicted vs experimental porosity")
plt.show()

#%%

fig = plt.figure(figsize=(5, 4), dpi=300)
scores = (r"$R^2={:.2f}$").format(
            round(r2_score(data7.iloc[:,-1:], 100-245.44*(0.8-data7["Pore size"])**2), 2),
        )

plt.scatter(data7.iloc[:,-1:],data7.iloc[:,:1]*reg.coef_+reg.intercept_,alpha=0.5,label="Machine Learning")
plt.scatter(data7.iloc[:,-1:],100-245.44*(0.8-data7["Pore size"])**2,color="C1",label="Explicit")
plt.plot([35, 70], [35, 70],color="Red")

plt.annotate(scores, xy=(40,180), xycoords='axes points',
            size=14,
            bbox=dict(boxstyle='round', fc='w'))


plt.legend(loc="lower right")
plt.xlabel("Measured porosity [%]")
plt.ylabel("Predicted porosity [%]")
plt.title("Predicted vs experimental porosity")
plt.show()
#%%
from scipy import optimize

#Def of the function for Hf
def linfunc(x1, A):
    fun = A*x1
    return fun

X = data7.iloc[:,-1].to_numpy()
Y = (data7.iloc[:,:1]*reg.coef_+reg.intercept_).iloc[:,0].to_numpy()

p_opt, p_cov = optimize.curve_fit(linfunc, X, Y, p0=[1])

func = linfunc(X, *p_opt)

residuals = Y - func
ss_res = np.sum(residuals**2)
ss_tot = np.sum((Y-np.mean(Y))**2)
r_squared = 1 - (ss_res / ss_tot)

print(*p_opt)

fig = plt.figure(figsize=(5, 4), dpi=300)
scores = (r"$R^2={:.2f}$").format(
            round(r_squared, 2),
        )
plt.scatter(X,Y)
plt.plot([35, 70], [35*p_opt[0], 70*p_opt[0]],color="Red")
plt.plot([35, 70], [35, 70],color="Orange")

plt.annotate(scores, xy=(40,180), xycoords='axes points',
            size=14,
            bbox=dict(boxstyle='round', fc='w'))

plt.xlabel("Measured porosity [%]")
plt.ylabel("Predicted porosity [%]")
plt.title("Predicted vs experimental porosity")
plt.show()


#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression

x = data7["Pore size"].to_numpy()
y = data7["Porosity"].to_numpy()

x_train = []
y_train = []
x_test = np.arange(x.min(), x.max() + 1).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0.3, 0.45)
ax.set_ylim(30, 70)
ax.set_xlabel('Pore Size [mm]')
ax.set_ylabel('Porosity [%]')
ax.grid()

scatter, = ax.plot([], [], 'go')
line, = ax.plot([], [], 'r')

lr = LinearRegression()


def animate(n):
    x_train.append([x[n]])
    y_train.append([y[n]])

    lr.fit(x_train, y_train)
    y_test = lr.predict(x_test)

    scatter.set_data(x_train, y_train)
    line.set_data(x_test, y_test)


anim = FuncAnimation(fig, animate, frames=x.size, interval=100, repeat=False)

f = r"C:\Users\cojo\OneDrive - Universidad Adolfo Ibanez\2022\Jose_2\Data\test.gif" 
writergif = animation.PillowWriter(fps=30) 
anim.save(f, writer=writergif)

#%%
def box_plot(data):
    bp = plt.boxplot(
                data,
                patch_artist = True, notch ='True'
                )
     
    colors = ['white', 'white','white','white']
     
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
     
    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='black',
                    linewidth = 1.5,
                    linestyle =":")
     
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='black',
                linewidth = 2)
     
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='black',
                   linewidth = 3)
     
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
                  color ='black',
                  alpha = 0.5)

parameters = ['EMod','YStress','YStrain','Pore size']
y = parameters[3]
colum = "Material"

data_box = data.dropna()
plt.figure(figsize=(5,4), dpi=1200)
data_graph = [data_box[(data_box[colum] == "DCM") & (data_box["%BG"] == 0) & (data_box["Temperature"] == 90)][y].to_numpy(),
              data_box[(data_box[colum] == "DCM") & (data_box["%BG"] == 0) & (data_box["Temperature"] == 110)][y].to_numpy(),
              data_box[(data_box[colum] == "DCM") & (data_box["%BG"] == 0) & (data_box["Temperature"] == 130)][y].to_numpy(),
              data_box[(data_box[colum] == "DCM") & (data_box["%BG"] == 10) & (data_box["Temperature"] == 130)][y].to_numpy(),
              data_box[data_box[colum] == "DCM"][y].to_numpy(),
              data_box[data_box[colum] == "ACE"][y].to_numpy(),
              data_box[data_box[colum] == "MINI"][y].to_numpy()
              ]
box_plot(data_graph)

plt.xticks([1, 2, 3], ['0', '10', '20'])
plt.xlabel("%BG")
plt.ylabel(y)
plt.show()

#%% Por hist scatter
i = -1
for _num, (_mat, _temp, _conc) in enumerate(data2.index.unique()):
    i = i+1
    aux = data[
        (data["Material"] == _mat) &
        (data["Temperature"] == int(_temp)) &
        (data["%BG"] == int(_conc))
    ].dropna()
    
    aux_mean = aux["Pore size"].mean()
    aux_std = aux["Pore size"].std()
    
    _space1 = "  " if _conc == 0 else ""
    _space2 = " " if _mat in ("PCL", "ACE") else ""
    _space3 = "  " if _temp == 90 else ""
    text = f"{_mat}{_space2} {_space1}{_conc}% {_space3}{_temp}C"
    
    plt.scatter(aux_mean,text)
    
plt.plot([.4,.4],[0,19],":",color="Red")
#plt.xlim(0,0.5)

#%% Por hist scatter

paper_info2 = pd.DataFrame()
for _num, (_mat, _temp, _conc) in enumerate(data2.index.unique()):
    aux = paper_info[
        (paper_info["Material"] == _mat) &
        (paper_info["Temperature"] == int(_temp)) &
        (paper_info["%BG"] == int(_conc))
    ].dropna()
    
    paper_info2 = paper_info2.append(aux.iloc[0],ignore_index=True)
    
plt.scatter(paper_info2["Pore"],paper_info2["Freq"])
plt.show()

graph_data = paper_info2[paper_info2["%BG"] == 0].groupby("Temperature").mean().reset_index()

graph_data_mean = paper_info2[
                                (paper_info2["%BG"] == 0)
                              ].groupby("Temperature").mean().reset_index()
graph_data_std = paper_info2[
                                (paper_info2["%BG"] == 0)
                              ].groupby("Temperature").std().reset_index()

graph_data_std = graph_data_std*1.96/np.sqrt(4)

#%%
# for _temp in TEMP:
# _aux = data[data["Temperature"] == _temp]
# _aux = _aux.groupby("Layer")

data["Circularity"] = data["Area"]*4*np.pi/data["Perimeter"]**2

_aux = data.groupby("Layer")
data_avg = _aux.mean().reset_index()
data_std = _aux.std().reset_index()
data_count = _aux.count().reset_index()
data_avg["Vertical print"] = data_avg["Vertical print"].round().astype(int)
data_std["Vertical print"] = data_avg["Vertical print"]
data_count["Vertical print"] = data_avg["Vertical print"]

fig, ax = plt.subplots(dpi=200)
ax.set_ylim(0, 1)
ax.set_xlim(2, 57)
_var = "Circularity"

direction = 0
_marker = "o" if direction else "^"
_label = "Average"

_avg = data_avg.reset_index()
_std = data_std.reset_index()
_count = data_count.reset_index()

#Smooth of curve
kernel_size = 2
kernel = np.ones(kernel_size) / kernel_size
_avg_fix = np.convolve(_avg[_var], kernel, mode='same')

_avg_fix[0] = _avg_fix[0]*2

# Mean
_color = "orange" if direction else "royalblue"
#ax.scatter(_avg["Layer"], _avg[_var], s=50, linewidths=0.25, zorder=100,
           #marker=_marker, color=_color, label=_label, edgecolors="black")

# STD
_color = "#e90" if direction else "#16e"
ax.plot(_avg["Layer"], _avg_fix, color="Black",label="Mean")
ax.fill_between(_avg["Layer"], _avg_fix - _std[_var],
                _avg_fix + _std[_var],label="SD",
            color="black",alpha=0.3
               )

"""
# STD fill between
_color = "#e90" if direction else "#16e"
for _sign in (1.0, -1.0):
    ax.fill_between(_avg["Layer"], _avg[_var] - _std[_var]*_sign,  _avg[_var] + _std[_var]*_sign,
               zorder=100, color=_color, linewidths=0.75,
               edgecolors=None, alpha=0.2
               )
    
# STD
_color = "#e90" if direction else "#16e"
for _sign in (1.0, -1.0):
    ax.scatter(_avg["Layer"], _avg[_var] + _std[_var]*_sign,
               zorder=100, color=_color, linewidths=0.75,
               edgecolors=None, s=30, marker="_",
               )
    

# 10th and 90th percentiles
_color = "#e90" if direction else "#16e"
for _q in (0.1, 0.9):
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore")
        _aux2 = _aux.quantile(_q).reset_index()
    _aux2 = _aux2[_aux2["Vertical print"] == direction]
    ax.scatter(_aux2["Layer"], _aux2[_var], s=10, zorder=100,
               marker="o", color=_color, facecolors="none", linewidths=0.75)


# Density
_data = data[data["Vertical print"] == direction].reset_index()
_data[_var] = _data[_var] + np.random.uniform(-0.005, 0.005, len(_data))
_data["Layer"] = _data["Layer"].astype(float) + np.random.uniform(-0.5, 0.5, len(_data))
_color = "#fc5" if direction else "#5af"
_data = _data.iloc[::10]
ax.scatter(_data["Layer"], _data[_var], alpha=1/20, s=4, zorder=10,
           marker=_marker, color=_color, edgecolors=None)
"""

ax.plot([0, 60], [0.785, 0.785], color="red", zorder=10, linewidth=2, linestyle=":",alpha=0.8)
# ax.set_title(f"Pore size vs layer - {_temp}C")
ax.set_xlabel("Number of Layer")
ax.set_ylabel("Circularity")
ax.grid(alpha=0.5, zorder=50)
plt.legend(loc="lower right")
plt.show()
