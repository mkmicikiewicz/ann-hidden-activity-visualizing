from network.data_loader import *
from network.network import *
from analysis.network_analysis import *
from scenarios.scenarios import *


def main():
    X_train, Y_train, X_test, Y_test = load_data_cnn(DataType.MNIST)

    # model = create_multilayer_perceptron(DataType.MNIST)
    # train_model(model, 16, 100, "mnist_mlp", X_train, Y_train)
    # load_weights_from_file(model, "mnist_mlp", 5, 5)
    # model_cnn = create_cnn(DataType.MNIST)
    # load_weights_from_file(model_cnn, "mnist_cnn", 100, 1)
    # train_model(model_cnn, 32, 100, "mnist_cnn", X_train, Y_train)

    model_bt = create_cnn(DataType.MNIST)
    model = load_model_from_file("mnist_cnn", 100)
    # network_history = load_network_history_from_file("svhn_cnn", 100)
    # plot_history(network_history)

    l1_bt, l2_bt = get_activations_cnn(model_bt, X_test)
    Y_predicted_bt = predict_classes(model_bt, X_test)
    transformed_points_bt, targets_bt = show_tsne("mnist_cnn_test_l2", 0, l2_bt[:2000], Y_test[:2000], Y_predicted_bt[:2000])
    x_bt = create_neuron_projection(l2_bt[:2000])
    plot_neuron_projection(x_bt, Y_test[:2000])
    #to_do next [11] extremely rand trees


    l1, l2 = get_activations_cnn(model, X_test)
    Y_predicted = predict_classes(model, X_test)
    transformed_points, targets = show_tsne("mnist_cnn_test_l2", 100, l2[:2000], Y_test[:2000], Y_predicted[:2000])
    x = create_neuron_projection(l2[:2000])
    plot_neuron_projection(x, Y_test[:2000])
    # to_do next [11] extremely rand trees


    # print(get_knn_accuracy(transformed_points, Y_test[:2000]))
    # count_power_modulo()
    # mnist_test_subset_cnn_last_hidden_layer_during_training_tsne_and_nh()

if __name__ == '__main__':
    main()
