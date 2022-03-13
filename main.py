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
import random
random.seed(42)


def put_features_in_device(input_features, device):
    for key in input_features.keys():
        if isinstance(input_features[key], Tensor):
            input_features[key] = input_features[key].to(device)


def eval_model(clustering_model, eval_texts, true_labels=None):
    eval_emb = clustering_model.qp_model.encode(eval_texts, convert_to_tensor=True)
    pred_labels = clustering_model.get_clustering(eval_emb, 3)
    rand_score = -1
    if true_labels is not None:
        rand_score = adjusted_rand_score(true_labels, pred_labels)
    return rand_score, eval_emb, pred_labels


def train_model(train_file, val_file, emb_model_name, emb_dim, device, model_out, batch_size=64, max_num_tokens=256, weight_decay=0.01,
                num_epochs=3, lrate=1e-5, warmup=1000, max_grad_norm=1.0):
    train_data = get_intent_data(train_file)
    val_data = get_intent_data(val_file)
    val_texts = [dat.split('_')[0] for dat in val_data]
    val_true_labels = [dat.split('_')[1] for dat in val_data]
    random.shuffle(train_data)
    num_batches = len(train_data) // batch_size
    num_train_steps = num_epochs * num_batches
    model = ClusteringModel(emb_model_name, emb_dim, device, max_num_tokens)
    model_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_train_steps)
    best_loss = 9999.0
    for epoch in range(num_epochs):
        for b in range(num_batches):
            batch_data = train_data[b * batch_size: (b+1) * batch_size]
            batch_texts = [dat.split('_')[0] for dat in batch_data]
            batch_labels = [dat.split('_')[1] for dat in batch_data]
            if len(set(batch_labels)) < 3:
                continue
            batch_features = model.qp_model.tokenize(batch_texts)
            gt = torch.zeros(batch_size, batch_size, device=device)
            gt_weights = torch.ones(batch_size, batch_size, device=device)
            para_label_freq = {k: batch_labels.count(k) for k in set(batch_labels)}
            for i in range(batch_size):
                for j in range(batch_size):
                    if batch_labels[i] == batch_labels[j]:
                        gt[i][j] = 1.0
                        gt_weights[i][j] = para_label_freq[batch_labels[i]]
            put_features_in_device(batch_features, device)
            mc, ma = model(batch_features, 3)
            sim_mat = 1 / (1 + torch.cdist(ma, ma))
            loss = torch.sum(((gt - sim_mat) ** 2) * gt_weights) / gt.shape[0]
            loss.backward()
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model, model_out)
                print('\nSaved model with loss %.4f\n' % best_loss)
            eval_score, _, _ = eval_model(model, val_texts, val_true_labels)
            print('\rLoss: %.4f, val adj RAND: %.4f' % (loss.item(), eval_score), end='')
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            opt.zero_grad()
            schd.step()


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    train_model('C:\\Users\\suman\\Documents\\niket_intent_work\\data\\train.jsonl',
                'C:\\Users\\suman\\Documents\\niket_intent_work\\data\\val.jsonl',
                'sentence-transformers/all-MiniLM-L6-v2',
                256,
                device,
                'saved_models/best_loss_clustering_model.model')


if __name__ == '__main__':
    main()