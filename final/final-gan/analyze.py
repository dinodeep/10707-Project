import torch
import torchvision

from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
import torch.utils.data as data

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import pickle

from networks import (
    MLP_CGAN_RELU,
    SMALL_MLP_CGAN_LEAKY,
    MLP_CGAN_LEAKY,
    BIG_MLP_CGAN_LEAKY,
    DCCGAN,
    Discriminator,
    LATENT_DIM
)
from train_gan import DEVICE, save_predictions
from datasets import load_fashion_original, load_fashion_imbalanced, load_fashion_upsampled, load_fashion_downsampled

plt.rcParams.update({"font.size": 22})

NUM_PREDICTIONS = 100
LOW_IMBALANCE_SCALE = 7/9
HIGH_IMBALANCE_SCALE = 1/2

ALL_DATASETS = ["original", "low_unbalanced", "high_unbalanced"]

ALL_GANS = {
    "mlp_cgan_relu": MLP_CGAN_RELU,
    "small_mlp_cgan_leaky": SMALL_MLP_CGAN_LEAKY,
    "mlp_cgan_leaky": MLP_CGAN_LEAKY,
    "big_mlp_cgan_leaky": BIG_MLP_CGAN_LEAKY,
    # "mlp_cgan_diff_loss": None,
    # "dccgan_leaky": DCCGAN,
}

ALL_RESULTS = [
    "original",
    "low_unbalanced",
    "high_unbalanced",
    "downsampled",
    "upsampled",
    "mlp_cgan_relu",
    "small_mlp_cgan_leaky",
    "mlp_cgan_leaky",
    "big_mlp_cgan_leaky",
]

CLASS2STRING = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}


'''
    - final results table for all gans
'''


def gen_gan_imgs():
    ''' 10 x 10 image of generated images for all gans (10 per class) '''
    folder = "writeup_results/gen_gan_imgs"
    os.makedirs(folder, exist_ok=True)
    for dataset in ALL_DATASETS:
        for gan_str, gencls in ALL_GANS.items():
            prefix = f"{dataset}_{gan_str}"
            path = f"results_cgan/{prefix}_generator.pt"
            G = gencls()
            G.load_state_dict(torch.load(path, map_location=DEVICE))
            G.eval()

            zs = torch.randn(10 * 10, LATENT_DIM)
            labels = torch.arange(0, 10).repeat_interleave(10).to(torch.int64)
            imgs = G(zs, labels)
            imgs = (imgs + 1) / 2

            save_path = os.path.join(folder, prefix + "_samples.png")
            save_predictions(G, zs, labels, save_path)

    return


def acc_loss_plots(methods, dataset, name="experiment-1.png"):
    ''' create plots of metrics w.r.t. CNN performance and baselines '''
    folder = "writeup_results/acc_loss_plots"
    os.makedirs(folder, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    for method in methods:
        # load and place on a plot
        filepath = f"results_cnn/{dataset}_{method}_results.pkl"
        with open(filepath, "rb") as f:
            d = pickle.load(f)

        epochs = np.arange(0, len(d["train_losses"]))
        axes[0].plot(epochs, d["test_losses"], label=f"{method}")
        axes[1].plot(epochs, d["test_accs"], label=f"{method}")
        axes[2].plot(epochs, d["test_f1s"], label=f"{method}")

    # add labeling and things
    axes[0].set_title(f"[train-dataset={dataset}] Test Losses")
    axes[0].set_xlabel(f"Epoch #")
    axes[0].set_ylabel(f"Test Loss")

    axes[1].set_title(f"[train-dataset={dataset}] Test Accs")
    axes[1].set_xlabel(f"Epoch #")
    axes[1].set_ylabel(f"Test Accs")

    axes[2].set_title(f"[train-dataset={dataset}] Test F1 Scores")
    axes[2].set_xlabel(f"Epoch #")
    axes[2].set_ylabel(f"Test F1 Scores")

    plt.legend()
    plt.savefig(os.path.join(folder, name))

    return


def plot_class_distributions(ratio=1, upsampled=False, downsampled=False, upsampling_method="", name=""):
    ''' class distribution for low imbalance and high imbalance '''
    # original, imbalanced_light, imbalanced_heavy, subsampled, upsampled, gan sampled
    folder = "writeup_results/data_distribution"
    os.makedirs(folder, exist_ok=True)

    max_count = 6000
    plt.figure(figsize=(20, 10))
    plt.gca().set_ylim([0, max_count])

    x = np.arange(0, 10)
    names = [CLASS2STRING[i] for i in x]
    bar_width = 1

    counts = [max_count * ratio ** i for i in x]
    if not downsampled:
        plt.bar(x, counts, color="#c9daf8", edgecolor='white', width=bar_width, label="real samples")
    else:
        plt.bar(x, [max(100, min(counts)) for c in counts], color="#c9daf8",
                edgecolor='white', width=bar_width, label="real samples")

    if upsampled:
        leftover_counts = [max_count - c for c in counts]
        plt.bar(x, leftover_counts, bottom=counts, color="#f9cb9c",
                edgecolor='white', width=bar_width, label=upsampling_method)

    plt.xticks(x, names)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.legend()

    plt.savefig(os.path.join(folder, name))

    return


def main():
    # gen_gan_imgs()

    # skipping the high degree dataset
    acc_loss_plots(ALL_RESULTS[0:2] + ALL_RESULTS[3:6], "high_unbalanced", name="high-experiment-1.png")
    acc_loss_plots(ALL_RESULTS[0:2] + ALL_RESULTS[3:9], "high_unbalanced", name="high-experiment-2.png")

    # class distributions for low and high classes
    # plot_class_distributions(ratio=1, upsampled=False, upsampling_method="", name="original-dataset.png")
    # plot_class_distributions(ratio=LOW_IMBALANCE_SCALE, upsampled=False,
    #                          upsampling_method="", name="low-imbalanced-dataset.png")
    # plot_class_distributions(ratio=HIGH_IMBALANCE_SCALE, upsampled=False,
    #                          upsampling_method="", name="high-imbalanced-dataset.png")

    # plot_class_distributions(ratio=LOW_IMBALANCE_SCALE, upsampled=True,
    #                          upsampling_method="Copy or GAN", name="low-upsampled.png")
    # plot_class_distributions(ratio=HIGH_IMBALANCE_SCALE, upsampled=True,
    #                          upsampling_method="Copy or GAN", name="high-upsampled.png")

    # plot_class_distributions(ratio=LOW_IMBALANCE_SCALE, downsampled=True,
    #                          upsampling_method="", name="low-downsampled.png")
    # plot_class_distributions(ratio=HIGH_IMBALANCE_SCALE, downsampled=True,
    #                          upsampling_method="", name="high-downsampled.png")


if __name__ == "__main__":
    main()
