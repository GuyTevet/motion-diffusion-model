import torch
import numpy as np


#adapted from action2motion
def calculate_diversity(activations):
    diversity_times = 200
    num_motions = len(activations)

    diversity = 0

    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times
    return diversity

# from action2motion
def calculate_diversity_multimodality(activations, labels, num_labels, unconstrained = False):
    diversity_times = 200
    multimodality_times = 20
    if not unconstrained:
        labels = labels.long()
    num_motions = activations.shape[0]  # len(labels)

    diversity = 0
        
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times

    if not unconstrained:
        multimodality = 0
        label_quotas = np.zeros(num_labels)
        label_quotas[labels.unique()] = multimodality_times  # if a label does not appear in batch, its quota remains zero
        while np.any(label_quotas > 0):
            # print(label_quotas)
            first_idx = np.random.randint(0, num_motions)
            first_label = labels[first_idx]
            if not label_quotas[first_label]:
                continue

            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]
            while first_label != second_label:
                second_idx = np.random.randint(0, num_motions)
                second_label = labels[second_idx]

            label_quotas[first_label] -= 1

            first_activation = activations[first_idx, :]
            second_activation = activations[second_idx, :]
            multimodality += torch.dist(first_activation,
                                        second_activation)

        multimodality /= (multimodality_times * num_labels)
    else:
        multimodality = torch.tensor(np.nan)

    return diversity.item(), multimodality.item()

