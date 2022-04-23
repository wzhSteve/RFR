# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

def get_data(transformed_features, type_mask, in_dims):
    data = transformed_features
    label = np.zeros((transformed_features.shape[0],), dtype=np.int)
    n_samples, n_features = data.shape
    color = [i for i in range(transformed_features.shape[0])]
    color_list = ['r', 'g', 'y', 'b', 'c', 'm']
    for i in range(len(in_dims)):
        node_indices = np.where(type_mask == i)[0]
        label[node_indices] = i
        for t in node_indices:
            color[t] = color_list[i]
    return data, label, n_samples, n_features, color


def plot_embedding(data, label, title):
    #scalar
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(label[i]),
    #              color=plt.cm.Set1(label[i] / 10.),
    #              fontdict={'weight': 'bold', 'size': 9})
    #plt.xticks([])
    #plt.yticks([])
    plt.title(title)
    return fig


def plot_tsne(embeddings, label):
    color_list = ['#66ff66', '#33ccff', '#336699', '#006699', 'c', 'm']
    color = [color_list[label[i]] for i in range(embeddings.shape[0])]
    print('Computing t-SNE embedding')

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(embeddings)
    # scalar
    x_min, x_max = np.min(result, 0), np.max(result, 0)
    result = (result - x_min) / (x_max - x_min)
    plt.scatter(result[:, 0], result[:, 1], color=color, cmap=plt.cm.Spectral)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    #ax = plt.gca()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    plt.show()
    #plt.savefig('figure/' + flag + '/visualization' + str(epoch) + '.jpg')

