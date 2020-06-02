from network.data_loader import *
from network.network import *
from analysis.network_analysis import *
from scenarios.scenarios import *


def main():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.MNIST)
    model = create_multilayer_perceptron(DataType.MNIST)
    train_model(model, 16, 5, "mnist_mlp", X_train, Y_train)
    load_weights_from_file(model, "mnist_mlp", 5, 5)
    # model_cnn = create_cnn(DataType.SVHN)
    # train_model(model_cnn, 32, 100, "svhn_cnn", X_train, Y_train)
    # model = load_model_from_file("mnist_mlp", 5)
    # network_history = load_network_history_from_file("svhn_cnn", 100)
    # plot_history(network_history)
    l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    # print(get_knn_accuracy(transformed_points, Y_test[:2000]))
    Y_predicted = predict_classes(model, X_test)
    transformed_points = show_tsne("mnist_mlp_test_l4", 5, l4, Y_test[:2000], Y_predicted[:2000])
    # count_power_modulo()


if __name__ == '__main__':
    main()
