import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

class NeuralClassifier(nn.Module):
    def __init__(self, activation_func, layer_size_list, use_batch_norm):
        super(NeuralClassifier, self).__init__()

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        self.classification_head = self.get_classification_head(activation_func, layer_size_list, use_batch_norm)
        self.classification_head.to(device)

    @staticmethod
    def get_classification_head(activation_func, layer_size_list, use_batch_norm):
        if activation_func == 'relu':
            activation_func_class = nn.ReLU
        elif activation_func == 'sigmoid':
            activation_func_class = nn.Sigmoid
        elif activation_func == 'tanh':
            activation_func_class = nn.Tanh
        else:
            assert False

        input_size = 768

        layers = []
        cur_input_size = input_size
        for cur_output_size in layer_size_list:
            layers.append(nn.Linear(cur_input_size, cur_output_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(cur_output_size))
            layers.append(activation_func_class())
            cur_input_size = cur_output_size
        layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.classification_head(x)
    
    def predict(self, input_features):
        with torch.no_grad():
            output = self.classification_head(input_features)
        return torch.argmax(output)
    

class SVMClassifier:
    def __init__(self, svm_kernel, standardize_data):
        self.classifier = SVC(kernel=svm_kernel)
        self.standardize_data = standardize_data

    def fit(self, training_mat, label_mat):
        if self.standardize_data:
            self.scale = StandardScaler().fit(training_mat)
            training_mat = self.scale.transform(training_mat)
        self.classifier.fit(training_mat, label_mat)

    def predict(self, input_features):
        if self.standardize_data:
            input_features = self.scale.transform(input_features)
        return self.classifier.predict(input_features)


def create_classifier(classifier_config):
    if classifier_config.classifier_type == 'neural':
        return NeuralClassifier(
            classifier_config.activation_func,
            classifier_config.layer_size_list,
            classifier_config.use_batch_norm
            ).to(torch.device('cuda'))
    elif classifier_config.classifier_type == 'svm':
        return SVMClassifier(
            classifier_config.svm_kernel,
            classifier_config.standardize_data
            )
    elif classifier_config.classifier_type == 'linear_regression':
        return LinearRegression()
