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


def plot_history(histories, key):
    plt.style.use("ggplot")
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history[key],'--', label=name.title() + ' Val')
        plt.plot(history[key], color=val[0].get_color(), label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.savefig(key+"plotPIC.png")
    plt.close()

plot_history([('baseline', history)], 'loss')
plot_history([('baseline', history)], 'regression_loss')
plot_history([('baseline', history)], 'classification_loss')

plt.show()
