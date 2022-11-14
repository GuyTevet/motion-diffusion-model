import torch
import numpy as np
from .models import load_classifier, load_classifier_for_fid
from .accuracy import calculate_accuracy
from .fid import calculate_fid
from .diversity import calculate_diversity_multimodality


class A2MEvaluation:
    def __init__(self, device):
        dataset_opt = {"input_size_raw": 72, "joints_num": 24, "num_classes": 12}
        
        self.input_size_raw = dataset_opt["input_size_raw"]
        self.num_classes = dataset_opt["num_classes"]
        self.device = device
        
        self.gru_classifier_for_fid = load_classifier_for_fid(self.input_size_raw, self.num_classes, device).eval()
        self.gru_classifier = load_classifier(self.input_size_raw,  self.num_classes, device).eval()
        
    def compute_features(self, model, motionloader):
        # calculate_activations_labels function from action2motion
        activations = []
        labels = []
        with torch.no_grad():
            for idx, batch in enumerate(motionloader):
                activations.append(self.gru_classifier_for_fid(batch["output_xyz"], lengths=batch["lengths"]))
                if model.cond_mode != 'no_cond':
                    labels.append(batch["y"])
            activations = torch.cat(activations, dim=0)
            if model.cond_mode != 'no_cond':
                labels = torch.cat(labels, dim=0)
        return activations, labels

    @staticmethod
    def calculate_activation_statistics(activations):
        activations = activations.cpu().numpy()
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def evaluate(self, model, loaders):
        
        def print_logs(metric, key):
            print(f"Computing action2motion {metric} on the {key} loader ...")
            
        metrics = {}
        
        computedfeats = {}
        for key, loader in loaders.items():
            metric = "accuracy"
            print_logs(metric, key)
            mkey = f"{metric}_{key}"
            if model.cond_mode != 'no_cond':
                metrics[mkey], _ = calculate_accuracy(model, loader,
                                                  self.num_classes,
                                                  self.gru_classifier, self.device)
            else:
                metrics[mkey] = np.nan

            # features for diversity
            print_logs("features", key)
            feats, labels = self.compute_features(model, loader)
            print_logs("stats", key)
            stats = self.calculate_activation_statistics(feats)
            
            computedfeats[key] = {"feats": feats,
                                  "labels": labels,
                                  "stats": stats}

            print_logs("diversity", key)
            ret = calculate_diversity_multimodality(feats, labels, self.num_classes, unconstrained=(model.cond_mode=='no_cond'))
            metrics[f"diversity_{key}"], metrics[f"multimodality_{key}"] = ret
            
        # taking the stats of the ground truth and remove it from the computed feats
        gtstats = computedfeats["gt"]["stats"]
        # computing fid
        for key, loader in computedfeats.items():
            metric = "fid"
            mkey = f"{metric}_{key}"
            
            stats = computedfeats[key]["stats"]
            metrics[mkey] = float(calculate_fid(gtstats, stats))
            
        return metrics
