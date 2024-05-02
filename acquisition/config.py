import os

cache_dir = 'cache_dir'
flickr_root_path = '/cs/labs/oabend/uriber/datasets/flickr30'
flickr_json_path = os.path.join(flickr_root_path, 'karpathy', 'dataset_flickr30k.json')
mcrae_path = '/cs/labs/oabend/uriber/datasets/mcrae_typicality.yaml'
swow_path = '/cs/labs/oabend/uriber/datasets/swow.en.csv'
class_to_pos_tag = [
    # Nouns:
    ['NN', 'NNS', 'NNP', 'WP', 'NNPS', 'WP$'],
    # Verbs:
    ['VBD', 'VB', 'VBP', 'VBG', 'VBZ', 'VBN', 'VERB'],
    # Adjectivs:
    ['JJ', 'JJR', 'JJS'],
    # Others:
    ['<unk>', 'UH', ',', 'PRP', 'PRP$', 'RB', '.', 'DT', 'O', 'IN', 'CD', 'WRB', 'WDT',
     'CC', 'TO', 'MD', ':', 'RP', 'EX', 'FW', 'XX', 'HYPH', 'POS', 'RBR', 'PDT', 'RBS',
     'AFX', '-LRB-', '-RRB-', '``', "''", 'LS', '$', 'SYM', 'ADD', '*', 'NFP']
]
pos_tag_to_class = {}
for class_ind in range(len(class_to_pos_tag)):
    for pos_tag in class_to_pos_tag[class_ind]:
        pos_tag_to_class[pos_tag] = class_ind
ontonotes_pos_tags = ["XX", "``", "$", "''", "*", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX", "CC", "CD", "DT", "EX", "FW", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "VERB", "WDT", "WP", "WP$", "WRB"]
embed_dim = 768
subword_pooling = 'mean' # mean/last
