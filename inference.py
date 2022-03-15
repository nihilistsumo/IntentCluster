import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import transformers
from transformers import AdamW
from sentence_transformers import models, SentenceTransformer
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import adjusted_rand_score
import numpy as np
from handle_data import get_intent_data
from clustering import ClusteringModel
from tqdm import tqdm
import json
import random
random.seed(42)


def get_best_documents(clustering_model_path, train_file, test_file, output_file):
    model = torch.load(clustering_model_path)
    train_data = get_intent_data(train_file)
    test_data = get_intent_data(test_file)
    test_questions = [dat.split('_')[0] for dat in test_data]
    train_questions = [dat.split('_')[0] for dat in train_data]
    test_emb = model.qp_model.encode(test_questions)
    train_emb = model.qp_model.encode(train_questions)
    with open(output_file, 'w', encoding='utf8') as f:
        for i in tqdm(range(len(test_questions))):
            closest_train_question = train_questions[np.argmin(np.linalg.norm(np.tile(test_emb[i],
                                                                    (len(train_questions), 1)) - train_emb, axis=1))]
            f.write(json.dumps({'test': test_questions[i], 'train': closest_train_question}))


def get_best_documents_simplified(clustering_model_path, train_text, query_text, output_text):
    model = torch.load(clustering_model_path)
    docs, queries = [], []
    with open(train_text, 'r', encoding='utf8') as f:
        for l in f:
            docs.append(l)
    with open(query_text, 'r', encoding='utf8') as f:
        for l in f:
            queries.append(l)
    query_emb = model.qp_model.encode(queries)
    doc_emb = model.qp_model.encode(docs)
    with open(output_text, 'w', encoding='utf8') as f:
        for i in tqdm(range(len(queries))):
            closest_doc = docs[np.argmin(np.linalg.norm(np.tile(query_emb[i], (len(docs), 1)) - doc_emb, axis=1))]
            f.write(closest_doc)


def main():
    '''
    get_best_documents('saved_models\\best_loss_clustering_model.model',
                       'C:\\Users\\suman\\Documents\\niket_intent_work\\data\\train.jsonl',
                       'C:\\Users\\suman\\Documents\\niket_intent_work\\data\\test.jsonl',
                       'infer_out\\best_loss_clustering_model.model.jsonl')
    '''
    get_best_documents_simplified('saved_models\\best_loss_soft_clustering_model.model',
                                  'C:\\Users\\suman\\Documents\\niket_intent_work\\query_label_data\\corpus.txt',
                                  'C:\\Users\\suman\\Documents\\niket_intent_work\\query_label_data\\queries.txt',
                                  'infer_out\\query_label_answer.txt')


if __name__ == '__main__':
    main()