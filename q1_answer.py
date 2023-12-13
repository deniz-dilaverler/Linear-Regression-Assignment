import numpy as np
import gzip
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from q1_script import read_pixels, read_labels
from pca import PcaHelper

images = read_pixels("data/train-images-idx3-ubyte.gz")
labels = read_labels("data/train-labels-idx1-ubyte.gz")


def question_1_1():
    print("Question 1.1")
    k = 10

    pca_helper = PcaHelper(images)
    pves = pca_helper.get_PVEs(k=k)

    print(f"The first {k} PVEs:")
    for i, pve in enumerate(pves):
        print(f"The PC{i + 1}'s: {pve:.5f}")


def question_1_2():
    print("Question 1.2")
    pca_helper = PcaHelper(images)
    cumulative_pves = pca_helper.get_cumulative_PVEs()

    print("The cumulative proportion of variances")
    for i, cumulative_pve in enumerate(cumulative_pves):
        print(
            f"The first {i + 1} PCs' cumulative proportion of  variances: {cumulative_pve:.5f}"
        )
        if cumulative_pve >= 0.7:
            break


def question_1_3():
    print("Question 1.3")
    k = 10
    pca_helper = PcaHelper(images)
    pca_helper.plot_k_PCs(k=k)


def question_1_4():
    print("Question 1.4")
    pca_helper = PcaHelper(images)
    pca_helper.project_images_on_2PCs(images, labels)


def question_1_5():
    print("Question 1.5")
    pca_helper = PcaHelper(images)

    Ks = [1, 50, 100, 250, 500, 784]
    pca_helper.plot_multiple_reconstructions(images, Ks)


question_1_1()

question_1_2()

question_1_3()

question_1_4()

question_1_5()
