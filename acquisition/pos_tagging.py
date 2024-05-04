from acquisition.collect_flickr_data import collect_flickr_data
from acquisition.embed import generate_features
from acquisition.config import flickr_json_path, pos_tag_to_class, ontonotes_pos_tags
from acquisition.generate_pos_data import generate_pos_data
from acquisition.classifier import create_classifier
from acquisition.trainer import create_trainer
from acquisition.classifier_config import ClassifierConfig
import random
import torch
from datasets import load_dataset

def get_ontonotes_data(split, binary):
    dataset = load_dataset('conll2012_ontonotesv5', 'english_v12')
    token_lists = []
    pos_data = []
    for doc in dataset[split]:
        for sentence_obj in doc['sentences']:
            token_lists.append(sentence_obj['words'])
            if binary:
                pos_data.append([
                    {
                        'text': word,
                        'label': 1 if pos_tag_to_class[ontonotes_pos_tags[pos_tag]] == 0 else 0
                    } for word, pos_tag in zip(sentence_obj['words'], sentence_obj['pos_tags'])
                ])
            else:
                pos_data.append([
                    {
                        'text': word,
                        'label': pos_tag_to_class[ontonotes_pos_tags[pos_tag]]
                    } for word, pos_tag in zip(sentence_obj['words'], sentence_obj['pos_tags'])
                ])

    return token_lists, pos_data

def get_data(model_path, binary, dataset):
    if dataset == 'flickr30k':
        sentences = collect_flickr_data(flickr_json_path, split='test')
        pos_data = generate_pos_data(sentences, binary, 'flickr30k')
        features = generate_features(model_path, sentences=sentences, tokens=None)
    elif dataset == 'ontonotes':
        tokens, pos_data = get_ontonotes_data(split='test', binary=binary)
        features = generate_features(model_path, sentences=None, tokens=tokens)

    assert len(features) == len(pos_data)

    # Features and pos data were created using different tokenizers, filter sentences that were tokenized differently
    data = []
    for feature_vectors, pos_data in zip(features, pos_data):
        if feature_vectors is None or torch.sum(torch.isnan(feature_vectors)) > 0 or feature_vectors.shape[0] != len(pos_data):
            continue
        data += [(feature_vectors[i], pos_data[i]['label']) for i in range(len(pos_data))]

    random.shuffle(data)
    train_sample_num = int(0.8*len(data))
    train_data = data[:train_sample_num]
    test_data = data[train_sample_num:]

    return train_data, test_data

def train_classifier(model_path, classifier_config, binary, dataset):
    train_data, test_data = get_data(model_path, binary, dataset)
    classifier = create_classifier(classifier_config)
    trainer = create_trainer(classifier, classifier_config, train_data, test_data)
    trainer.train()
    accuracy, res_mat = trainer.evaluate()
    return accuracy, res_mat

def run_pos_tagging_experiment(noise_images, version, dataset):
    config = ClassifierConfig()
    config.classifier_type = 'svm'
    noise_images_str = ''
    if noise_images:
        noise_images_str = '_noise_images'
    model_path = f'output/Pretrain{noise_images_str}/checkpoint_0{version}.pth'
    return train_classifier(model_path, config, True, dataset)
