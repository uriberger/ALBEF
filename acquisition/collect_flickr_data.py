import json

def collect_flickr_data(json_path, split=None):
    with open(json_path, 'r') as fp:
        data = json.load(fp)['images']
    
    if split is None:
        sentences = [x['raw'] for outer in data for x in outer['sentences']]
    else:
        sentences = [x['raw'] for outer in data for x in outer['sentences'] if outer['split'] == split]

    return sentences
