class ClassifierConfig:
    def __init__(self):
        self.classifier_type = 'neural'
        # Nerual related attributes
        self.activation_func = 'relu' # 'relu', 'sigmoid' or 'tanh'
        self.layer_size_list = [4]
        self.use_batch_norm = False
        # SVM related attributes
        self.svm_kernel = 'rbf'
        self.standardize_data = False
