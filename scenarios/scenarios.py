from network.data_loader import *
from network.network import *
from analysis.network_analysis import *


def mnist_test_subset_raw_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.MNIST)
    transformed_points = show_tsne("mnist_raw_test", 10000, X_test, Y_test)
    print(get_knn_accuracy(transformed_points, Y_test))


def mnist_test_subset_mlp_last_hidden_layer_before_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.MNIST)
    model = create_multilayer_perceptron(DataType.MNIST)
    l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    transformed_points = show_tsne("mnist_mlp_test_l4", 0, l4, Y_test[:2000])
    print(get_knn_accuracy(transformed_points, Y_test[:2000]))


def mnist_test_subset_mlp_last_hidden_layer_after_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.MNIST)
    model = load_model_from_file("mnist_mlp", 100)
    l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    Y_predicted = predict_classes(model, X_test)
    transformed_points = show_tsne("mnist_mlp_test_l4", 100, l4, Y_test[:2000], Y_predicted[:2000])
    print(get_knn_accuracy(transformed_points, Y_test[:2000]))


def svhn_test_subset_mlp_last_hidden_layer_before_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.SVHN)
    model = create_multilayer_perceptron(DataType.SVHN)
    l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    transformed_points = show_tsne("svhn_mlp_test_l4", 0, l4, Y_test[:2000])
    print(get_knn_accuracy(transformed_points, Y_test[:2000]))


def svhn_test_subset_mlp_first_hidden_layer_after_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.SVHN)
    model = load_model_from_file("svhn_mlp", 100)
    l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    Y_predicted = predict_classes(model, X_test)
    transformed_points = show_tsne("svhn_mlp_test_l1", 100, l1, Y_test[:2000], Y_predicted[:2000])
    print(get_knn_accuracy(transformed_points, Y_test[:2000]))


def svhn_test_subset_mlp_last_hidden_layer_after_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.SVHN)
    model = load_model_from_file("svhn_mlp", 100)
    l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    Y_predicted = predict_classes(model, X_test)
    transformed_points = show_tsne("svhn_mlp_test_l4", 100, l4, Y_test[:2000], Y_predicted[:2000])
    print(get_knn_accuracy(transformed_points, Y_test[:2000]))


def svhn_train_subset_mlp_last_hidden_layer_after_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.SVHN)
    model = load_model_from_file("svhn_mlp", 100)
    l1, l2, l3, l4 = get_activations_mlp(model, X_train)
    Y_predicted = predict_classes(model, X_train)
    transformed_points = show_tsne("svhn_mlp_train_l4", 100, l4, Y_train[:2000], Y_predicted[:2000])
    print(get_knn_accuracy(transformed_points, Y_train[:2000]))


def cifar_test_subset_cnn_last_hidden_layer_after_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_cnn(DataType.CIFAR)
    model = load_model_from_file("cifar_cnn", 100)
    l1, l2 = get_activations_cnn(model, X_test)
    Y_predicted = predict_classes(model, X_test)
    transformed_points = show_tsne("cifar_cnn_test_l2", 100, l2, Y_test[:2000], Y_predicted[:2000])
    print(get_knn_accuracy(transformed_points, Y_test[:2000]))
