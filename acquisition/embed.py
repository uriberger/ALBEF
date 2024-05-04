from models.model_pretrain import ALBEF
from models.tokenization_bert import BertTokenizer
import ruamel.yaml as yaml
import torch
from tqdm import tqdm
import math

def load_model(model_path):
    device = torch.device('cuda')
    my_yaml = yaml.YAML(typ='rt')
    config = my_yaml.load(open('configs/Pretrain.yaml', 'r'))
    text_encoder = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer, init_deit=True)
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location='cpu') 
    state_dict = checkpoint['model']    
    model.load_state_dict(state_dict)

    return model

def agg_vectors(vectors, method):
    if method == 'mean':
        return torch.mean(vectors, dim=0)
    elif method == 'first':
        return vectors[0]
    elif method == 'last':
        return vectors[-1]
    else:
        assert False, f'Unknown feature aggregation method: {method}'

def extract_features_from_sentences(sentences, model, agg_subtokens_method):
    with torch.no_grad():
        text = model.tokenizer(sentences, padding=True, return_tensors="pt").to(torch.device('cuda'))
        text_feats = model.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True, mode = 'text', output_hidden_states = True).hidden_states[-1]

    feature_list = []
    for sent_ind in range(len(sentences)):
        cur_token_start_ind = None
        feature_vectors = []
        for i, text_id in enumerate(text['input_ids'][sent_ind]):
            if text_id.item() == 101:
                continue
            if text_id.item() == 102:
                break
            id_str = model.tokenizer.decode(text_id).replace(' ', '')
            if id_str.startswith('##'):
                continue
            if id_str == "'" and i < len(text['input_ids'][sent_ind]) - 1 and model.tokenizer.decode(text['input_ids'][sent_ind][i+1]) == 's':
                continue
            if i < len(text['input_ids'][sent_ind]) - 1 and model.tokenizer.decode(text['input_ids'][sent_ind][i+1]) == '-':
                continue
            if id_str == '-':
                continue
            elif cur_token_start_ind is not None:
                feature_vector = agg_vectors(text_feats[sent_ind, cur_token_start_ind:i, :], agg_subtokens_method)
                feature_vectors.append(feature_vector)
            cur_token_start_ind = i
        feature_vector = agg_vectors(text_feats[sent_ind, cur_token_start_ind:i+1, :], agg_subtokens_method)
        feature_vectors.append(feature_vector)

        feature_vectors = [x.unsqueeze(dim=0) for x in feature_vectors]
        feature_list.append(torch.cat(feature_vectors, dim=0))
        
    return feature_list

def extract_features_from_tokens(token_lists, model, agg_subtokens_method):
    sentences = [' '.join(x) for x in token_lists]
    with torch.no_grad():
        text = model.tokenizer(sentences, padding=True, return_tensors="pt").to(torch.device('cuda'))
        text_feats = model.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True, mode = 'text', output_hidden_states = True).hidden_states[-1]

    feature_list = []
    for sent_ind in range(len(sentences)):
        token_ind = 0
        cur_token = ''
        failed = False
        prev_token_end_ind = 0
        feature_vectors = []
        for i, text_id in enumerate(text['input_ids'][sent_ind]):
            if text_id.item() == 101:
                continue
            if text_id.item() == 102:
                break
            id_str = model.tokenizer.decode(text_id)
            if id_str.startswith('##'):
                cur_token += id_str[2:]
            else:
                cur_token += id_str

            if cur_token.lower() == token_lists[sent_ind][token_ind].lower():
                feature_vector = agg_vectors(text_feats[sent_ind, prev_token_end_ind+1:i+1, :], agg_subtokens_method)
                feature_vectors.append(feature_vector)
                prev_token_end_ind = i
                token_ind += 1
                cur_token = ''
            elif len(cur_token) > len(token_lists[sent_ind][token_ind]):
                assert cur_token.lower().startswith(token_lists[sent_ind][token_ind].lower()), f'Something wrong in the following sentence: {token_lists[sent_ind]} in token number {token_ind}, i.e. {token_lists[sent_ind][token_ind]}'
                feature_list.append(None)
                failed = True
                break

        if not failed:
            feature_vectors = [x.unsqueeze(dim=0) for x in feature_vectors]
            feature_list.append(torch.cat(feature_vectors, dim=0))

    return feature_list

def generate_features(model_path, sentences=None, tokens=None):
    assert bool(sentences is None) != bool(tokens is None), f'Exactly one of sentences and tokens should be none'
    res = []

    print('Loading model...', flush=True)
    model = load_model(model_path)

    # Batches
    batch_size = 4
    first_batch = len(res)
    sample_num = len(sentences) if sentences is not None else len(tokens)
    batch_num = math.ceil(sample_num/batch_size)

    for batch_ind in tqdm(range(first_batch, batch_num)):
        batch_start = batch_ind * batch_size
        batch_end = min((batch_ind + 1) * batch_size, sample_num)
        if sentences is not None:
            batch = sentences[batch_start:batch_end]
            res += extract_features_from_sentences(batch, model, agg_subtokens_method='mean')
        else:
            batch = tokens[batch_start:batch_end]
            res += extract_features_from_tokens(batch, model, agg_subtokens_method='mean')

    return res
