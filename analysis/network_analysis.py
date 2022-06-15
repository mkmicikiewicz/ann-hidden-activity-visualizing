from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras import backend as K
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from network.network import *
from network.data_loader import *
from network.constants import TSNE_PATH_PREFIX
from datashader.bundling import hammer_bundle
import networkx as nx


def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history['loss'])
    plt.plot(network_history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(network_history['accuracy'])
    plt.plot(network_history['val_accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()


def get_activations_mlp(model, data, size):
    layer_output = K.function([model.layers[0].input],
                              [model.layers[0].output, model.layers[2].output, model.layers[4].output,
                               model.layers[6].output])

    return layer_output([data[:size, :]])


def get_activations_cnn(model, data, size=None):
    layer_output = K.function([model.layers[0].input], [
                              model.layers[9].output, model.layers[10].output])
    if size:
        return layer_output([data[:size, :]])
    return layer_output([data[:2000, :]])


def show_tsne(model_name, epochs, X, Y, Y_predicted=None, init=None):
    data = StandardScaler().fit_transform(X)
    targets = np.argmax(Y, axis=1)

    file_path = TSNE_PATH_PREFIX + model_name + "_" + str(epochs) + ".npy"
    # if exists(file_path):
    #     points_transformed = np.load(file_path)
    # else:
    if init is not None:
        points_transformed = TSNE(n_components=2, perplexity=30, init=init,
                                  random_state=np.random.RandomState(0)).fit_transform(data).T

    else:
        points_transformed = TSNE(n_components=2, perplexity=30,
                                  random_state=np.random.RandomState(0)).fit_transform(data).T
    points_transformed = np.swapaxes(points_transformed, 0, 1)
    np.save(file_path, points_transformed)
    show_scatterplot(points_transformed, targets, Y_predicted)

    return points_transformed, targets


def show_scatterplot(points_transformed, targets, Y_predicted=None):
    if not Y_predicted:
        palette = sns.color_palette("bright", 10)
        plt.figure(figsize=(10, 10))
        sns.scatterplot(points_transformed[:, 0], points_transformed[:,
                        1], hue=targets, legend='full', palette=palette)
        plt.show()
    else:
        Y_diff = targets - Y_predicted
        styles = np.empty(shape=[0, 1], dtype=str)
        for y in Y_diff:
            if y == 0:
                styles = np.append(styles, 'Matched')
            else:
                styles = np.append(styles, 'Mismatched')

        palette = sns.color_palette("bright", 10)
        plt.figure(figsize=(10, 10))
        sns.scatterplot(points_transformed[:, 0], points_transformed[:, 1], hue=targets, style=styles,
                        legend='full', palette=palette)
        plt.show()


def get_knn_accuracy(X, Y):
    knn = NearestNeighbors(n_neighbors=7)
    knn.fit(X)

    ratio = 0
    Y = np.argmax(Y, axis=1)
    for neighbors in knn.kneighbors(X, return_distance=False):
        count = 0
        for neighbor in neighbors[1:]:
            if Y[neighbors[0]] == Y[neighbor]:
                count += 1
        ratio += count / 6

    return ratio / X.shape[0]


def plot_neuron_projection(x, Y):
    # plt.figure(figsize=(5, 5))
    # plt.scatter(x[:, 0], x[:, 1])
    # plt.show()
    # targets = np.argmax(Y, axis=1)
    plt.figure(figsize=(10, 10))
    # palette = sns.color_palette("bright", 10)
    print('plotting', x.shape)
    print(x)
    sns.scatterplot(x[:, 0], x[:, 1])
    plt.show()


# --------------------------------------------------------

def plot_new_neuron_projection(x, hue=None):
    palette = sns.color_palette("magma_r", as_cmap=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=hue, ax=ax, palette=palette)


def compare_projections(datatype, model_name, n_layer, digit):
    size = 2000

    if 'mlp' in model_name:
        X_train, Y_train, X_test, Y_test = load_data_mlp(datatype)
        model_bt = create_multilayer_perceptron(datatype)
        layer_bt = get_activations_mlp(model_bt, X_test)[n_layer - 1]
        x_bt = create_neuron_projection(layer_bt[:size])

        model_at = create_multilayer_perceptron(datatype)
        load_weights_from_file(model_at, model_name, 100, 100)
        layer_at = get_activations_mlp(model_at, X_test)[n_layer - 1]
        x_at = create_neuron_projection(layer_at[:size])
    elif 'cnn' in model_name:
        X_train, Y_train, X_test, Y_test = load_data_cnn(datatype)
        model_bt = create_cnn(datatype)
        layer_bt = get_activations_cnn(model_bt, X_test)[n_layer - 1]
        x_bt = create_neuron_projection(layer_bt[:size])

        model_at = create_cnn(datatype)
        load_weights_from_file(model_at, model_name, 100, 100)
        layer_at = get_activations_cnn(model_at, X_test)[n_layer - 1]
        x_at = create_neuron_projection(layer_at[:size])

    digit_test = np.argmax(Y_test[:size], axis=1) == digit
    etc_bt = ExtraTreesClassifier()
    etc_bt.fit(layer_bt, digit_test)
    etc_at = ExtraTreesClassifier()
    etc_at.fit(layer_at, digit_test)

    plot_new_neuron_projection(x_bt, hue=etc_bt.feature_importances_)
    plot_new_neuron_projection(x_at, hue=etc_at.feature_importances_)


def plot_discriminative_map(activations, Y_test, size):
    palette = sns.color_palette('bright', 10)
    lst = []
    for digit in range(10):
        digit_test = np.argmax(Y_test[:size], axis=1) == digit
        etc = ExtraTreesClassifier()
        etc.fit(activations, digit_test)
        lst.append(etc.feature_importances_)
    arr = np.dstack(lst)
    labels = np.argmax(arr, axis=2).flatten()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    projection = create_neuron_projection(activations)
    def norm(x): return (x - np.min(x, axis=1)) / \
        (np.max(x, axis=1) - np.min(x, axis=1))
    for n in range(10):
        projection_part = projection[labels == n]
        saturation = norm(arr[:, labels == n, n])
        sns.scatterplot(x=projection_part[:, 0], y=projection_part[:, 1],
                        alpha=saturation, ax=ax, label=n, palette=palette)


def compare_discriminative_map(datatype, model_name, n_layer, size):
    if 'mlp' in model_name:
        X_train, Y_train, X_test, Y_test = load_data_mlp(datatype)
        model_bt = create_multilayer_perceptron(datatype)
        layer_bt = get_activations_mlp(model_bt, X_test)[n_layer - 1]

        model_at = create_multilayer_perceptron(datatype)
        load_weights_from_file(model_at, model_name, 100, 100)
        layer_at = get_activations_mlp(model_at, X_test)[n_layer - 1]
    elif 'cnn' in model_name:
        X_train, Y_train, X_test, Y_test = load_data_cnn(datatype)
        model_bt = create_cnn(datatype)
        layer_bt = get_activations_cnn(model_bt, X_test)[n_layer - 1]

        model_at = create_cnn(datatype)
        load_weights_from_file(model_at, model_name, 100, 100)
        layer_at = get_activations_cnn(model_at, X_test)[n_layer - 1]
    plot_discriminative_map(layer_bt, Y_test, size)
    plot_discriminative_map(layer_at, Y_test, size)


def show_seq_projections(datatype, model_name, n_layer, epoch, size):
    init = None
    if 'mlp' in model_name:
        X_train, Y_train, X_test, Y_test = load_data_mlp(datatype)
        model_at = create_multilayer_perceptron(datatype)
        load_weights_from_file(model_at, model_name, 100, epoch)
        layer_at = get_activations_mlp(model_at, X_test, size)[n_layer]
    elif 'cnn' in model_name:
        X_train, Y_train, X_test, Y_test = load_data_cnn(datatype)
        model_at = create_cnn(datatype)
        load_weights_from_file(model_at, model_name, 100, epoch)
        layer_at = get_activations_cnn(model_at, X_test, size)[n_layer]
    file_path = TSNE_PATH_PREFIX + model_name + "_" + str(epoch) + ".npy"
    if os.path.exists(file_path):
        init = np.load(file_path)
    points_transformed, targets = show_tsne(
        model_name, epoch, layer_at, Y_test[:size], init=init)
    return points_transformed, targets


def inter_layer_evolution(points_lst, targets):
    points_by_digit = []
    for n in range(10):
        points_by_layer = []
        for points in points_lst:
            ind = targets == n
            points_by_layer.append(points[ind])
        points_by_digit.append(points_by_layer)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    for label, epoch_points in enumerate(points_by_digit):
        activation_lst = []
        for n in range(epoch_points[0].shape[0]):
            layer_lst = []
            for count, layer in enumerate(epoch_points):
                layer_lst.append(epoch_points[count][n, :])
            activation_lst.append(layer_lst)
        dfs_lst = []
        for activation_count, lst in enumerate(activation_lst, 1):
            dct = {}
            for count, value in enumerate(lst, 1):
                dct[f'activation_{activation_count}_layer_{count}'] = value
            df = pd.DataFrame(dct)
            dfs_lst.append(df)
        graph = nx.Graph()
        for df in dfs_lst:
            df_dense = df.corr('pearson')
            for edge, _ in df_dense.unstack().items():
                if int(edge[0].split('_')[-1]) + 1 == int(edge[1].split('_')[-1]):
                    graph.add_edge(*edge)
        c_dct = dict()
        for layer_count, layer_value in enumerate(epoch_points, 1):
            for activation_count, activation_value in enumerate(layer_value, 1):
                c_dct[f'activation_{activation_count}_layer_{layer_count}'] = activation_value
        complete_df = pd.DataFrame(c_dct).T
        complete_df.reset_index(inplace=True)
        nodes = complete_df.rename(columns={'index': 'name', 0: 'x', 1: 'y'})
        sources = list()
        targets = list()
        for source, target in list(graph.edges):
            sources.append(nodes[nodes['name'] == source].index[0])
            targets.append(nodes[nodes['name'] == target].index[0])
        edges = pd.DataFrame({'source': sources, 'target': targets})
        hb = hammer_bundle(nodes, edges)
        hb.plot(x="x", y="y", figsize=(10, 10), ax=ax, alpha=0.7, label=label)


def get_all_activations(datatype, model_name, n_layer, size):
    activations = []
    for epoch in [0, 20, 40, 60, 80, 100]:
        if 'mlp' in model_name:
            X_train, Y_train, X_test, Y_test = load_data_mlp(datatype)
            model_at = create_multilayer_perceptron(datatype)
            if epoch:
                load_weights_from_file(model_at, model_name, 100, epoch)
            layer_at = get_activations_mlp(model_at, X_test, size)[n_layer-1]
        elif 'cnn' in model_name:
            X_train, Y_train, X_test, Y_test = load_data_cnn(datatype)
            model_at = create_cnn(datatype)
            if epoch:
                load_weights_from_file(model_at, model_name, 100, epoch)
            layer_at = get_activations_cnn(model_at, X_test, size)[n_layer-1]
        activations.append(layer_at)
    return activations, Y_test[:size]


def show_whole_tsne(X, Y):
    data = StandardScaler().fit_transform(X)
    targets = np.argmax(Y, axis=1)
    points_transformed = TSNE(n_components=2, perplexity=30,
                              random_state=np.random.RandomState(0)).fit_transform(data).T
    points_transformed = np.swapaxes(points_transformed, 0, 1)
    show_scatterplot(points_transformed, targets)
    return points_transformed, targets


def process_activations(activations, Y_test, size):
    arr_activations = np.concatenate(activations, axis=0)
    arr_targets = np.concatenate([Y_test for _ in range(6)], axis=0)
    points_transformed, targets = show_whole_tsne(arr_activations, arr_targets)
    points_lst = list(points_transformed.reshape(6, size, 2))
    return points_lst, targets[:size]


def inter_epoch_evolution(points_lst, targets):
    points_by_digit = []
    for n in range(10):
        points_by_layer = []
        for points in points_lst:
            ind = targets == n
            points_by_layer.append(points[ind])
        points_by_digit.append(points_by_layer)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    for label, epoch_points in enumerate(points_by_digit):
        activation_lst = []
        for n in range(epoch_points[0].shape[0]):
            epoch_lst = []
            for count, epoch in enumerate(epoch_points):
                epoch_lst.append(epoch_points[count][n, :])
            activation_lst.append(epoch_lst)
        dfs_lst = []
        for activation_count, lst in enumerate(activation_lst, 1):
            dct = {}
            for count, value in enumerate(lst):
                dct[f'activation_{activation_count}_epoch_{count * 20}'] = value
            df = pd.DataFrame(dct)
            dfs_lst.append(df)
        graph = nx.Graph()
        for df in dfs_lst:
            df_dense = df.corr('pearson')
            for edge, _ in df_dense.unstack().items():
                if int(edge[0].split('_')[-1]) + 20 == int(edge[1].split('_')[-1]):
                    graph.add_edge(*edge)
        c_dct = dict()
        for epoch_count, epoch_value in enumerate(epoch_points):
            for activation_count, activation_value in enumerate(epoch_value, 1):
                c_dct[f'activation_{activation_count}_epoch_{epoch_count * 20}'] = activation_value
        complete_df = pd.DataFrame(c_dct).T
        complete_df.reset_index(inplace=True)
        nodes = complete_df.rename(columns={'index': 'name', 0: 'x', 1: 'y'})
        sources = list()
        targets = list()
        for source, target in list(graph.edges):
            sources.append(nodes[nodes['name'] == source].index[0])
            targets.append(nodes[nodes['name'] == target].index[0])
        edges = pd.DataFrame({'source': sources, 'target': targets})
        hb = hammer_bundle(nodes, edges)
        hb.plot(x="x", y="y", figsize=(10, 10), ax=ax, alpha=0.7, label=label)
# --------------------------------------------------------
# inter-epoch evolution
# --------------------------------------------------------


def get_tsne(model_name, epochs, X, Y, init=None):
    data = StandardScaler().fit_transform(X)
    targets = np.argmax(Y, axis=1)

    file_path = TSNE_PATH_PREFIX + model_name + "_" + str(epochs) + ".npy"
    if exists(file_path):
        points_transformed = np.load(file_path)
    else:
        if init is not None:
            points_transformed = TSNE(n_components=2, perplexity=30, init=init,
                                      random_state=np.random.RandomState(0)).fit_transform(data).T
            np.save(file_path, points_transformed)
        else:
            points_transformed = TSNE(n_components=2, perplexity=30,
                                      random_state=np.random.RandomState(0)).fit_transform(data).T
            np.save(file_path, points_transformed)
    points_transformed = np.swapaxes(points_transformed, 0, 1)

    return points_transformed, targets


def show_tsne_epoch_trace(model_name, datatype):
    epoch_points_transformed = []

    if 'mlp' in model_name:
        _, _, X_test, Y_test = load_data_mlp(datatype)

        def create_model():
            return create_multilayer_perceptron(datatype)

        def get_activations(m, d):
            _, _, _, l4 = get_activations_mlp(m, d)
            return l4
    elif 'cnn' in model_name:
        _, _, X_test, Y_test = load_data_cnn(datatype)

        def create_model():
            return create_cnn(datatype)

        def get_activations(m, d):
            _, l2 = get_activations_cnn(m, d)
            return l2

    targets = np.argmax(Y_test[:2000], axis=1)

    # for untrained model
    model = create_model()
    l = get_activations(model, X_test)
    Y_predicted = predict_classes(model, X_test)
    points_transformed, _ = show_tsne(
        model_name + "_last_layer", 0, l, Y_test[:2000], Y_predicted[:2000])

    epoch_points_transformed += [points_transformed]

    for epochs in [20, 40, 60, 80, 100]:
        model = create_model()
        load_weights_from_file(model, model_name, 100, epochs)
        l = get_activations(model, X_test)
        Y_predicted = predict_classes(model, X_test)
        points_transformed, _ = show_tsne(
            model_name + "_last_layer", epochs, l, Y_test[:2000], Y_predicted[:2000])

        epoch_points_transformed += [points_transformed]

    show_trace(epoch_points_transformed, targets)


def show_trace(points, targets):
    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    palette = sns.color_palette("bright", 10)
    plt.figure(figsize=(10, 10))
    # sns.scatterplot(points_transformed[:, 0], points_transformed[:,
    #                 1], hue=targets, legend='full', palette=palette)

    for i in range(len(points[0][:, 0])):
        for p in range(len(points)-1):
            cmap = cmap = plt.get_cmap(cmaps[targets[i]])
            xs = [points[p][i][0], points[p+1][i][0]]
            ys = [points[p][i][1], points[p+1][i][1]]
            plt.plot(xs, ys, c=cmap(p+1/len(points)))
    plt.show()


# --------------------------------------------------------
# inter-layer evolution
# --------------------------------------------------------

def show_tsne_layer_trace(model_name, datatype, show_last=False):
    epoch_points_transformed = []

    _, _, X_test, Y_test = load_data_mlp(datatype)

    model = create_multilayer_perceptron(datatype)
    load_weights_from_file(model, model_name, 100, 100)
    l1, l2, l3, l4 = get_activations_mlp(model,  X_test)

    Y_predicted = predict_classes(model, X_test)
    targets = np.argmax(Y_test[:2000], axis=1)

    layers = [l1, l2, l3]
    if show_last:
        layers += [l4]

    initial_points, _ = show_tsne(model_name + "_l1", 100, layers[0], Y_test[:2000],
                                  Y_predicted[:2000])
    epoch_points_transformed = [initial_points]

    for index, layer in enumerate(layers[1:]):
        transformed_points, _ = show_tsne(model_name + "_l" + str(index + 2), 100, layer, Y_test[:2000],
                                          Y_predicted[:2000], initial_points)
        epoch_points_transformed += [transformed_points]

    show_trace(epoch_points_transformed, targets)
