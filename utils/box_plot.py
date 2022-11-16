import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

key, value = [], []
import pandas as pd

df_ERM = pd.read_excel('Reliable SSL.xlsx', "ERM-CIFAR10-1")
df_Context= pd.read_excel('Reliable SSL.xlsx', "Context-CIFAR10-1")
df_Rotation = pd.read_excel('Reliable SSL.xlsx', "Rotation-CIFAR10-1")
df_Affine = pd.read_excel('Reliable SSL.xlsx', "Affine-CIFAR10-1")
df_Jigsaw = pd.read_excel('Reliable SSL.xlsx', "Jigsaw-CIFAR10-1")

def get_accuracies(df, dname):
    out = []
    for i in range(0, 40, 4):
        out.append(df[dname][i])
    return out

# dname = "iid"
# dname = "brightness"
# dname = "gaussian_noise"
dname = "zoom_blur"
key += ["ERM"] * 10
value += get_accuracies(df_ERM, dname)
key += ["Context"] * 10
value += get_accuracies(df_Context, dname)
key += ["Rotation"] * 10
value += get_accuracies(df_Rotation, dname)
key += ["Affine"] * 10
value += get_accuracies(df_Affine, dname)
key += ["Jigsaw"] * 10
value += get_accuracies(df_Jigsaw, dname)


d = {'key': key, 'value': value}
df = pd.DataFrame(data=d)

sns.boxplot(x=df["key"], y=df["value"])

plt.xlabel("Method")
plt.ylabel('Accuracy')

plt.title(dname)
plt.savefig(dname + "_acc.png")