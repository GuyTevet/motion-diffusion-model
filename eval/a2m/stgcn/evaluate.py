import torch
import numpy as np
from .accuracy import calculate_accuracy
from .fid import calculate_fid
from .diversity import calculate_diversity_multimodality

from eval.a2m.recognition.models.stgcn import STGCN


class Evaluation:
    def __init__(self, dataname, parameters, device, seed=None):
        layout = "smpl"  # if parameters["glob"] else "smpl_noglobal"
        model = STGCN(in_channels=parameters["nfeats"],
                      num_class=parameters["num_classes"],
                      graph_args={"layout": layout, "strategy": "spatial"},
                      edge_importance_weighting=True,
                      device=device)

        model = model.to(device)

        model_path = "./assets/actionrecognition/uestc_rot6d_stgcn.tar"

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        self.num_classes = parameters["num_classes"]
        self.model = model

        self.dataname = dataname
        self.device = device

        self.seed = seed

    def compute_features(self, model, motionloader):
        # calculate_activations_labels function from action2motion
        activations = []
        labels = []
        with torch.no_grad():
            for idx, batch in enumerate(motionloader):
                activations.append(self.model(batch)["features"])
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
            print(f"Computing stgcn {metric} on the {key} loader ...")

        metrics_all = {}
        for sets in ["train", "test"]:
            computedfeats = {}
            metrics = {}
            for key, loaderSets in loaders.items():
                loader = loaderSets[sets]

                metric = "accuracy"
                mkey = f"{metric}_{key}"
                if model.cond_mode != 'no_cond':
                    print_logs(metric, key)
                    metrics[mkey], _ = calculate_accuracy(model, loader,
                                                          self.num_classes,
                                                          self.model, self.device)
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
                ret = calculate_diversity_multimodality(feats, labels, self.num_classes,
                                                        seed=self.seed, unconstrained=(model.cond_mode=='no_cond'))
                metrics[f"diversity_{key}"], metrics[f"multimodality_{key}"] = ret

            # taking the stats of the ground truth and remove it from the computed feats
            gtstats = computedfeats["gt"]["stats"]
            # computing fid
            for key, loader in computedfeats.items():
                metric = "fid"
                mkey = f"{metric}_{key}"

                stats = computedfeats[key]["stats"]
                metrics[mkey] = float(calculate_fid(gtstats, stats))

            metrics_all[sets] = metrics

        metrics = {}
        for sets in ["train", "test"]:
            for key in metrics_all[sets]:
                metrics[f"{key}_{sets}"] = metrics_all[sets][key]
        return metrics
