import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_skewgraph(level):
    key, value, skew = [], [], []
    df_ERM = pd.read_excel("Reliable SSL.xlsx", "ERM-CIFAR10-" + str(level))
    df_Context = pd.read_excel("Reliable SSL.xlsx", "Context-CIFAR10-" + str(level))
    df_Rotation = pd.read_excel("Reliable SSL.xlsx", "Rotation-CIFAR10-" + str(level))
    df_Affine = pd.read_excel("Reliable SSL.xlsx", "Affine-CIFAR10-" + str(level))
    df_Jigsaw = pd.read_excel("Reliable SSL.xlsx", "Jigsaw-CIFAR10-" + str(level))

    def get_accuracies(df):
        out = []
        for i in range(3, 22):
            out.append(df.iloc[42][i])

        return out

    key += ["ERM"] * 19
    value += get_accuracies(df_ERM)
    skew += [level] * 19
    key += ["Context"] * 19
    value += get_accuracies(df_Context)
    skew += [level] * 19
    key += ["Rotation"] * 19
    value += get_accuracies(df_Rotation)
    skew += [level] * 19
    key += ["Affine"] * 19
    value += get_accuracies(df_Affine)
    skew += [level] * 19
    key += ["Jigsaw"] * 19
    value += get_accuracies(df_Jigsaw)
    skew += [level] * 19
    return key, value, skew


key0, value0, skew0 = ["ERM", "Context", "Rotation", "Affine", "Jigsaw"], [], ["Test", "Test", "Test", "Test", "Test"]

df_ERM = pd.read_excel("Reliable SSL.xlsx", "ERM-CIFAR10-" + str(1))
df_Context = pd.read_excel("Reliable SSL.xlsx", "Context-CIFAR10-" + str(1))
df_Rotation = pd.read_excel("Reliable SSL.xlsx", "Rotation-CIFAR10-" + str(1))
df_Affine = pd.read_excel("Reliable SSL.xlsx", "Affine-CIFAR10-" + str(1))
df_Jigsaw = pd.read_excel("Reliable SSL.xlsx", "Jigsaw-CIFAR10-" + str(1))
value0 = [
    df_ERM.iloc[42][2],
    df_Context.iloc[42][2],
    df_Rotation.iloc[42][2],
    df_Affine.iloc[42][2],
    df_Jigsaw.iloc[42][2],
]

key1, value1, skew1 = create_skewgraph(level=1)
key2, value2, skew2 = create_skewgraph(level=2)
key3, value3, skew3 = create_skewgraph(level=3)
key4, value4, skew4 = create_skewgraph(level=4)
key5, value5, skew5 = create_skewgraph(level=5)
d = {
    "Method": key0 + key1 + key2 + key3 + key4 + key5,
    "acc": value0 + value1 + value2 + value3 + value4 + value5,
    "skew": skew0 + skew1 + skew2 + skew3 + skew4 + skew5,
}
df = pd.DataFrame(data=d)

fig, ax1 = plt.subplots(figsize=(15, 5))
plt.rcParams.update({"font.size": 15})

sns.boxplot(data=df, x="skew", y="acc", hue="Method")

count = 0
for line in ax1.get_lines()[:5]:
    line.set_color("tab:blue")
    line.set_linewidth(2)
for line in ax1.get_lines()[10:15]:
    line.set_color("tab:orange")
    line.set_linewidth(2)
for line in ax1.get_lines()[15:20]:
    line.set_color("tab:green")
    line.set_linewidth(2)
for line in ax1.get_lines()[20:23]:
    line.set_color("tab:red")
    line.set_linewidth(2)
for line in ax1.get_lines()[27:30]:
    line.set_color("tab:purple")
    line.set_linewidth(2)

plt.xlabel("Shift Intensity", fontsize=20)
plt.ylabel("Negative Log-Likelihood", fontsize=20)
plt.tick_params(axis="both", which="major", labelsize=20)
plt.tight_layout()
plt.savefig("nll.pdf")
