from network.data_loader import *
from network.network import *
from analysis.network_analysis import *
from scenarios.scenarios import *


def main():
    # X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.SVHN)
    # model = create_multilayer_perceptron(DataType.SVHN)
    # train_model(model, 16, 100, "svhn_mlp", X_train, Y_train)
    # model_cnn = create_cnn(DataType.CIFAR)
    # train_model(model_cnn, 32, 100, "cifar_cnn", X_train, Y_train)
    # model = load_model_from_file("svhn_mlp", 100)
    # network_history = load_network_history_from_file("cifar_cnn", 100)
    # plot_history(network_history)
    # l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    # print(get_knn_accuracy(transformed_points, Y_test[:2000]))
    # svhn_test_subset_mlp_last_hidden_layer_before_training_tsne_and_nh()
    # mnist_test_subset_raw_tsne_and_nh()
    mnist_test_subset_mlp_last_hidden_layer_before_training_tsne_and_nh()
    # mnist_test_subset_mlp_last_hidden_layer_after_training_tsne_and_nh()
    # svhn_test_subset_mlp_last_hidden_layer_before_training_tsne_and_nh()
    # cifar_test_subset_cnn_last_hidden_layer_after_training_tsne_and_nh()
    # Y_predicted = predict_classes(model, X_test)
    # transformed_points = show_tsne("svhn_mlp_test_l4", 100, l4, Y_test[:2000], Y_predicted[:2000])


if __name__ == '__main__':
    main()
