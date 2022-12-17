import numpy as np
import matplotlib.pylab as plt
import pandas as pd

x = [0, 1, 2, 3, 4, 5]

def get_NLL(method):
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

def get_Acc(method):
    y_out, yerr_out = [], []
    df_iid = pd.read_excel('Reliable SSL.xlsx', method + "-CIFAR10-1")
    y_out.append(df_iid.iloc[:,2][40])
    yerr_out.append(df_iid.iloc[:,2][43])
    out = []
    for i in range(1, 6):
        df_iid = pd.read_excel('Reliable SSL.xlsx', method + "-CIFAR10-" + str(i))
        y_out.append(df_iid.iloc[:,22][40])
        yerr_out.append(df_iid.iloc[:,22][43])
   
    return y_out, yerr_out

def get_ECE(method):
    y_out, yerr_out = [], []
    df_iid = pd.read_excel('Reliable SSL.xlsx', method + "-CIFAR10-1")
    y_out.append(df_iid.iloc[:,2][41])
    yerr_out.append(df_iid.iloc[:,2][44])
    out = []
    for i in range(1, 6):
        df_iid = pd.read_excel('Reliable SSL.xlsx', method + "-CIFAR10-" + str(i))
        y_out.append(df_iid.iloc[:,22][41])
        yerr_out.append(df_iid.iloc[:,22][44])
   
    return y_out, yerr_out

y_erm, yerr_erm = get_NLL("ERM")
y_context, yerr_context = get_NLL("Context")
y_rotation, yerr_rotation = get_NLL("Rotation")
y_affine, yerr_affine = get_NLL("Affine")
y_js, yerr_js = get_NLL("Jigsaw")

a_erm, aerr_erm = get_Acc("ERM")
a_context, aerr_context = get_Acc("Context")
a_rotation, aerr_rotation = get_Acc("Rotation")
a_affine, aerr_affine = get_Acc("Affine")
a_js, aerr_js = get_Acc("Jigsaw")

e_erm, eerr_erm = get_ECE("ERM")
e_context, eerr_context = get_ECE("Context")
e_rotation, eerr_rotation = get_ECE("Rotation")
e_affine, eerr_affine = get_ECE("Affine")
e_js, eerr_js = get_ECE("Jigsaw")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.errorbar(x, y_erm, yerr = yerr_erm, capsize=5, label='ERM') 
ax1.errorbar(x, y_context, yerr = yerr_context, capsize=5, label='Context') 
ax1.errorbar(x, y_rotation, yerr = yerr_rotation, capsize=5, label='Rotation') 
ax1.errorbar(x, y_affine, yerr = yerr_affine, capsize=5, label='Affine') 
ax1.errorbar(x, y_js, yerr = yerr_js, capsize=5, label='Jigsaw') 
ax1.set(xlabel='Shift Intensity', ylabel='Negative Log-Likelihood')
ax1.set_xticks(x) 

ax2.errorbar(x, a_erm, yerr = aerr_erm, capsize=5, label='ERM') 
ax2.errorbar(x, a_context, yerr = aerr_context, capsize=5, label='Context') 
ax2.errorbar(x, a_rotation, yerr = aerr_rotation, capsize=5, label='Rotation') 
ax2.errorbar(x, a_affine, yerr = aerr_affine, capsize=5, label='Affine') 
ax2.errorbar(x, a_js, yerr = aerr_js, capsize=5, label='Jigsaw') 
ax2.set(xlabel='Shift Intensity', ylabel='Accuracy')
ax2.set_xticks(x) 


ax3.errorbar(x, e_erm, yerr = eerr_erm, capsize=5, label='ERM') 
ax3.errorbar(x, e_context, yerr = eerr_context, capsize=5, label='Context') 
ax3.errorbar(x, e_rotation, yerr = eerr_rotation, capsize=5, label='Rotation') 
ax3.errorbar(x, e_affine, yerr = eerr_affine, capsize=5, label='Affine') 
ax3.errorbar(x, e_js, yerr = eerr_js, capsize=5, label='Jigsaw') 
ax3.set(xlabel='Shift Intensity', ylabel='Expected Calibration Error')
ax3.set_xticks(x) 


# Put a legend below current axis
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=True, ncol=5)

plt.subplots_adjust(left=0.1,
                    bottom=0.3,
                    right=0.98,
                    top=0.7,
                    wspace=0.4,
                    hspace=0.4)
plt.savefig("out_nll.pdf",bbox_inches='tight')
