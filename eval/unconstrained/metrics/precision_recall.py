# based on https://github.com/blandocs/improved-precision-and-recall-metric-pytorch/blob/master/functions.py
import os, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# self.batch_size = args.batch_size
# self.cpu = args.cpu
# self.data_size = args.data_size

def precision_and_recall(generated_features, real_features):
    k = 3

    data_num = min(len(generated_features), len(real_features))
    print(f'data num: {data_num}')

    if data_num <= 0:
        print("there is no data")
        return
    generated_features = generated_features[:data_num]
    real_features = real_features[:data_num]

    # get precision and recall
    precision = manifold_estimate(real_features, generated_features, k)
    recall = manifold_estimate(generated_features, real_features, k)

    return precision, recall

def manifold_estimate( A_features, B_features, k):
    A_features = list(A_features)
    B_features = list(B_features)
    KNN_list_in_A = {}
    for A in tqdm(A_features, ncols=80):
        pairwise_distances = np.zeros(shape=(len(A_features)))

        for i, A_prime in enumerate(A_features):
            d = torch.norm((A - A_prime), 2)
            pairwise_distances[i] = d

        v = np.partition(pairwise_distances, k)[k]
        KNN_list_in_A[A] = v

    n = 0

    for B in tqdm(B_features, ncols=80):
        for A_prime in A_features:
            d = torch.norm((B - A_prime), 2)
            if d <= KNN_list_in_A[A_prime]:
                n += 1
                break

    return n / len(B_features)


