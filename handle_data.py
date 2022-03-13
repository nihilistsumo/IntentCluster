import json


def get_intent_data(intent_file):
    intent_data = []
    with open(intent_file, 'r') as f:
        for l in f:
            d = json.loads(l)
            label = d['label']
            q = d['question']
            intent_data.append(q+'_'+label)
    return intent_data