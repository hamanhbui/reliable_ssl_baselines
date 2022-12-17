import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


key, value = [], []
import pandas as pd


df_1 = pd.read_excel("Reliable SSL.xlsx", "Jigsaw-CIFAR10-1")
df_2 = pd.read_excel("Reliable SSL.xlsx", "Jigsaw-CIFAR10-2")
df_3 = pd.read_excel("Reliable SSL.xlsx", "Jigsaw-CIFAR10-3")
df_4 = pd.read_excel("Reliable SSL.xlsx", "Jigsaw-CIFAR10-4")
df_5 = pd.read_excel("Reliable SSL.xlsx", "Jigsaw-CIFAR10-5")

acc = [
    df_1["Unnamed: 22"][42],
    df_2["Unnamed: 22"][42],
    df_3["Unnamed: 22"][42],
    df_4["Unnamed: 22"][42],
    df_5["Unnamed: 22"][42],
]
acc = np.asarray(acc)
print(np.mean(acc))
