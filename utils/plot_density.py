import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
from scipy.stats import wasserstein_distance

def draw_plot(method, color):
    with open("utils/out/"+method+"/iid/te_entropies.pkl", "rb") as fp:
        iid_entropies = pickle.load(fp)
    with open("utils/out/"+method+"/skew/te_entropies.pkl", "rb") as fp:
        skew_entropies = pickle.load(fp)
    with open("utils/out/"+method+"/ood/te_entropies.pkl", "rb") as fp:
        ood_entropies = pickle.load(fp)

    # # seaborn histogram
    plt.xlabel("Predictive Entropy")
    plt.xlim(-0.2, 2)
    plt.ylim(0, 15)
    sns.kdeplot(iid_entropies,color=color, label="iid")
    sns.kdeplot(skew_entropies,color=color, label="skew", linestyle="-.")
    sns.kdeplot(ood_entropies,color=color, label="ood", linestyle="dotted")
    plt.title(method, size=15)

    plt.legend()
    plt.tight_layout()
    plt.savefig("utils/out/"+method+"/"+method+"_predictive_entropy.pdf")

# draw_plot("ERM", "tab:blue")
# draw_plot("Context", "tab:orange")
# draw_plot("Rotation", "tab:green")
# draw_plot("Affine", "tab:red")
draw_plot("Jigsaw", "tab:purple")


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
