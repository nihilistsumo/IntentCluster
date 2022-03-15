import json
from sentence_transformers import SentenceTransformer


def get_intent_data(intent_file):
    intent_data = []
    with open(intent_file, 'r') as f:
        for l in f:
            d = json.loads(l)
            label = d['label']
            q = d['question']
            intent_data.append(q+'_'+label)
    return intent_data


def get_query_label_data(query_label_file, sbert_model_name):
    with open(query_label_file, 'r') as f:
        query_label_data = json.load(f)
        queries = [d['query'] for d in query_label_data]
        labels = [d['label'] for d in query_label_data]
    model = SentenceTransformer(sbert_model_name)
    label_emb_vec = model.encode(labels)
    return queries, labels, label_emb_vec