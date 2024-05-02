from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F

class Trainer:
    def __init__(self, classifier, config, train_data, test_data):
        self.classifier = classifier
        self.config = config
        self.train_data = train_data
        self.test_data = test_data

class NeuralTrainer(Trainer):
    def __init__(self, classifier, train_data, test_data, config):
        super(NeuralTrainer, self).__init__(classifier, train_data, test_data, config)
        self.device = torch.device('cuda')

    def train(self):
        dataloader = DataLoader(self.train_data, batch_size=64, shuffle=True)
        criterion = F.cross_entropy
        optimizer = optim.Adam(self.classifier.parameters())
        checkpoint_len = 100

        for epoch_ind in range(10):
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.classifier(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % checkpoint_len == 0:
                    print(f'[{epoch_ind }, {i + 1:5d}] loss: {running_loss / checkpoint_len:.3f}')
                    running_loss = 0.0

    def evaluate(self):
        dataloader = DataLoader(self.test_data, batch_size=64)
        correct = 0
        class_num = self.config.layer_size_list[-1]
        res_mat = np.zeros((class_num, class_num))

        for data in tqdm(dataloader):
            inputs, labels = data

            with torch.no_grad():
                outputs = self.classifier(inputs)
            predicted_classes = torch.max(outputs, dim=1)[1]
            
            correct += len([i for i in range(len(labels)) if predicted_classes[i] == labels[i]])
            for i in range(len(labels)):
                res_mat[predicted_classes[i].item(), labels[i].item()] += 1

        return correct/len(dataloader), res_mat

class SVMTrainer(Trainer):
    def __init__(self, classifier, train_data, test_data, config):
        super(SVMTrainer, self).__init__(classifier, train_data, test_data, config)

    def train(self):
        X = np.stack([x[0].cpu() for x in self.train_data])
        y = np.array([x[1] for x in self.train_data])
        print('[SVM trainer] training...', flush=True)
        self.classifier.fit(X, y)
        print('[SVM trainer] Finished training', flush=True)

    def evaluate(self):
        X = np.stack([x[0].cpu() for x in self.test_data])
        y = np.array([x[1] for x in self.test_data])
        predicted = self.classifier.predict(X)
        correct = len([i for i in range(len(predicted)) if predicted[i] == y[i]])
        class_num = len(set(list(y)))
        res_mat = np.zeros((class_num, class_num))

        for pred, gt in zip(predicted, y):
            res_mat[pred, gt] += 1

        return correct/predicted.shape[0], res_mat
    
class LinearRegressionTrainer(Trainer):
    def __init__(self, classifier, train_data, test_data, config):
        super(LinearRegressionTrainer, self).__init__(classifier, train_data, test_data, config)

    def train(self):
        X = np.stack([x[0].cpu() for x in self.train_data])
        y = np.array([x[1] for x in self.train_data])
        print('[Linear regression trainer] training...', flush=True)
        self.classifier.fit(X, y)
        print('[Linear regression trainer] Finished training', flush=True)

    def evaluate(self):
        X = np.stack([x[0].cpu() for x in self.test_data])
        y = np.array([x[1] for x in self.test_data])
        predicted = self.classifier.predict(X)

        return np.mean(np.square(y-predicted))

def create_trainer(classifier, classifier_config, train_data, test_data):
    if classifier_config.classifier_type == 'neural':
        return NeuralTrainer(classifier, classifier_config, train_data, test_data)
    elif classifier_config.classifier_type == 'svm':
        return SVMTrainer(classifier, classifier_config, train_data, test_data)
    elif classifier_config.classifier_type == 'linear_regression':
        return LinearRegressionTrainer(classifier, classifier_config, train_data, test_data)
    else:
        assert False, 'Classifier ' + classifier_config.classifier_type + ' unknown'
