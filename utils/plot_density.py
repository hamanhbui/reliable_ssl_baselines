import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
from scipy.stats import wasserstein_distance

with open("../algorithms/ERM/results/plots/CIFAR10_1/te_entropies.pkl", "rb") as fp:
    test_ERM = pickle.load(fp)
with open("../algorithms/Affine/results/plots/CIFAR10_1/te_entropies.pkl", "rb") as fp:
    test_Affine = pickle.load(fp)
# with open("../algorithms/Context/results/plots/CIFAR10_1/te_entropies.pkl", "rb") as fp:
#     test_Context = pickle.load(fp)
with open("../algorithms/Jigsaw/results/plots/CIFAR10_1/te_entropies.pkl", "rb") as fp:
    test_Jigsaw = pickle.load(fp)
with open("../algorithms/Rotation/results/plots/CIFAR10_1/te_entropies.pkl", "rb") as fp:
    test_Rotation = pickle.load(fp)

# # seaborn histogram
plt.xlabel("Predictive Entropy")
sns.kdeplot(test_ERM, label="ERM")
sns.kdeplot(test_Affine, label="Affine")
# sns.kdeplot(test_Context, label="Context")
sns.kdeplot(test_Jigsaw, label="Jigsaw")
sns.kdeplot(test_Rotation, label="Rotation")
plt.title("Out of distribution - CIFAR10", size=20)

plt.legend()
plt.savefig("predictive_entropy.png")

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
