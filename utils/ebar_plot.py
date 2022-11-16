import numpy as np
import matplotlib.pylab as plt
import pandas as pd

x = [0, 1, 2, 3, 4, 5]

def get_data(method):
    y_out, yerr_out = [], []
    df_iid = pd.read_excel('Reliable SSL.xlsx', method + "-CIFAR10-1")
    y_out.append(df_iid.iloc[:,2][42])
    yerr_out.append(df_iid.iloc[:,2][45])
    out = []
    for i in range(1, 6):
        df_iid = pd.read_excel('Reliable SSL.xlsx', method + "-CIFAR10-" + str(i))
        y_out.append(df_iid.iloc[:,22][42])
        yerr_out.append(df_iid.iloc[:,22][45])
   
    return y_out, yerr_out

y_erm, yerr_erm = get_data("ERM")
y_js, yerr_js = get_data("Jigsaw")

plt.figure()
plt.errorbar(x, y_erm, yerr = yerr_erm, capsize=6, label='ERM') 
plt.errorbar(x, y_js, yerr = yerr_js, capsize=6, label='Jigsaw') 
plt.ylabel("NLL",  size=18)
plt.xlabel("Skew Intensity",  size=18)
plt.legend(loc='lower right')
plt.savefig("out_nll.png")
