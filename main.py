from network.data_loader import *
from network.network import *
from analysis.network_analysis import *


def main():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.MNIST)
    # model = create_multilayer_perceptron(DataType.MNIST)
    # train_model(model, 16, 3, "mnist_mlp", X_train, Y_train)
    # model_cnn = create_cnn(DataType.CIFAR)
    # train_model(model_cnn, 32, 1, "cifar_cnn", X_train, Y_train)
    # network_history = load_network_history_from_file("svhn_mlp", 3)
    # plot_history(network_history)
    # model = load_model_from_file("mnist_mlp", 3)
    # l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    transformed_points = show_tsne("mnist_raw_test", 10000, X_test, Y_test)
    print(get_knn_accuracy(transformed_points, Y_test))


if __name__ == '__main__':
    main()
