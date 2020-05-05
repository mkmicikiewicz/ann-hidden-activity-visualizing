## Visualizing the hidden activity of artificial neural networks
### Mateusz Knapik, Jakub Czerski 2020

#### Saved model and network history
The following format applies:
- for model: model_<dataset_name>_<network_type>_<epochs>
- for network history: model_<dataset_name>_<network_type>_<epochs>_history.json

#### Saved transformet points (t-SNE)
Each t-SNE transformation saves transformed points (if and only if such a transformed points does not exist).
The following format applies: points_<model_name>_<layer>_<epochs>.npy

#### SVHN dataset
These files should be placed in __data/__ directory:
- http://ufldl.stanford.edu/housenumbers/train_32x32.mat
- http://ufldl.stanford.edu/housenumbers/test_32x32.mat
