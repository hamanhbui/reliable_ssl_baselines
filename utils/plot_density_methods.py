import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
from scipy.stats import wasserstein_distance


with open("utils/out/ERM/ood/te_entropies.pkl", "rb") as fp:
    test_ERM = pickle.load(fp)
with open("utils/out/Context/ood/te_entropies.pkl", "rb") as fp:
    test_Context = pickle.load(fp)
with open("utils/out/Rotation/ood/te_entropies.pkl", "rb") as fp:
    test_Rotation = pickle.load(fp)
with open("utils/out/Affine/ood/te_entropies.pkl", "rb") as fp:
    test_Affine = pickle.load(fp)
with open("utils/out/Jigsaw/ood/te_entropies.pkl", "rb") as fp:
    test_Jigsaw = pickle.load(fp)

# # seaborn histogram
sns.kdeplot(test_ERM, label="ERM")
sns.kdeplot(test_Context, label="Context")
sns.kdeplot(test_Rotation, label="Rotation")
sns.kdeplot(test_Affine, label="Affine")
sns.kdeplot(test_Jigsaw, label="Jigsaw")
plt.xlabel("Predictive Entropy", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.title("Out of distribution - CIFAR-10.1 v6", size=15)

plt.legend()
plt.tight_layout()
plt.savefig("predictive_entropy.pdf")

# with open("../algorithms/VAE/results/plots/MNIST_5/tr_nlls.pkl", "rb") as fp:
#     train_NLL = pickle.load(fp)

# with open("../algorithms/VAE/results/plots/MNIST_5/te_nlls.pkl", "rb") as fp:
#     test_NLL = pickle.load(fp)

# with open("../algorithms/VAE/results/plots/MNIST_5/adapt_nlls.pkl", "rb") as fp:
#     adapted_NLL = pickle.load(fp)

# plt.figure(figsize=(20, 10))
# plt.xlabel("Log density")
# plt.hist(train_NLL, label="train", density=True, bins=int(180 / 5))
# plt.hist(test_NLL, label="test", density=True, bins=int(180 / 5))
# plt.hist(adapted_NLL, label="adapt", density=True, bins=int(180 / 5))
# plt.legend()
# plt.savefig("out/out.png")
