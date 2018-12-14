import os
import wget
import argparse
import pickle

# import keras
import keras

import sys
sys.path.append('.\\keras-retinanet')

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import cv2


with open('./trainHistoryDict', 'rb') as file_pi:
    history = pickle.load(file_pi)


def plot_history(history, keys):
    plt.style.use("ggplot")
    plt.figure(figsize=(16, 10))

    for key in keys:
        val = plt.plot(history[key],'--', label=key.title())

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.savefig("zout"+ keys[0]+ ".png")
    plt.close()

plot_history(history, ['loss', 'val_loss'])
plot_history(history, ['classification_loss', 'val_classification_loss', 'regression_loss', 'val_regression_loss'])

plt.show()
