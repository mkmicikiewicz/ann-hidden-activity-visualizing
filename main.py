from network.data_loader import *
from network.network import *
from analysis.network_analysis import *


def main():
    X_train, Y_train, x_test, y_test = load_data_mlp(DataType.SVHN)
    model = create_multilayer_perceptron(DataType.SVHN)
    train_model(model, 16, 3, "svhn_mlp", X_train, Y_train)
    # model_cnn = create_cnn(DataType.CIFAR)
    # train_model(model_cnn, 32, 1, "cifar_cnn", X_train, Y_train)
    network_history = load_network_history_from_file("svhn_mlp", 3)
    plot_history(network_history)
    # model = load_model_from_file("mnist_cnn", 10)
    # l1, l2 = get_activations_cnn(model, X_train)
    # show_tsne("mnist_cnn_l1", 10, l1, Y_train[:2000])


if __name__ == '__main__':
    main()
