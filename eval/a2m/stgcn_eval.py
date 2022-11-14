import copy
import torch
from tqdm import tqdm
import functools

from utils.fixseed import fixseed

from eval.a2m.stgcn.evaluate import Evaluation as STGCNEvaluation
from torch.utils.data import DataLoader
from data_loaders.tensors import collate


from .tools import format_metrics
import utils.rotation_conversions as geometry
from utils import dist_util


def convert_x_to_rot6d(x, pose_rep):
    # convert rotation to rot6d
    if pose_rep == "rotvec":
        x = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(x))
    elif pose_rep == "rotmat":
        x = x.reshape(*x.shape[:-1], 3, 3)
        x = geometry.matrix_to_rotation_6d(x)
    elif pose_rep == "rotquat":
        x = geometry.matrix_to_rotation_6d(geometry.quaternion_to_matrix(x))
    elif pose_rep == "rot6d":
        x = x
    else:
        raise NotImplementedError("No geometry for this one.")
    return x


class NewDataloader:
    def __init__(self, mode, model, diffusion, dataiterator, device, cond_mode, dataset, num_samples):
        assert mode in ["gen", "gt"]

        self.batches = []
        sample_fn = diffusion.p_sample_loop

        with torch.no_grad():
            for motions, model_kwargs in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                motions = motions.to(device)
                if num_samples != -1 and len(self.batches) * dataiterator.batch_size > num_samples:
                    continue  # do not break because it confuses the multiple loaders
                batch = dict()
                if mode == "gen":
                    sample = sample_fn(model, motions.shape, clip_denoised=False, model_kwargs=model_kwargs)
                    batch['output'] = sample
                elif mode == "gt":
                    batch['output'] = motions

                max_n_frames = model_kwargs['y']['lengths'].max()
                mask = model_kwargs['y']['mask'].reshape(dataiterator.batch_size, max_n_frames).bool()
                batch["output_xyz"] = model.rot2xyz(x=batch["output"], mask=mask, pose_rep='rot6d', glob=True,
                                                    translation=True, jointstype='smpl', vertstrans=True, betas=None,
                                                    beta=0, glob_rot=None, get_rotations_back=False)
                if model.translation:
                    # the stgcn model expects rotations only
                    batch["output"] = batch["output"][:, :-1]

                batch["lengths"] = model_kwargs['y']['lengths'].to(device)
                # using torch.long so lengths/action will be used as indices
                if cond_mode != 'no_cond':  # proceed only if not running unconstrained
                    batch["y"] = model_kwargs['y']['action'].squeeze().long().cpu()  # using torch.long so lengths/action will be used as indices
                self.batches.append(batch)

            num_samples_last_batch = num_samples % dataiterator.batch_size
            if num_samples_last_batch > 0:
                for k, v in self.batches[-1].items():
                    self.batches[-1][k] = v[:num_samples_last_batch]


    def __iter__(self):
        return iter(self.batches)


def evaluate(args, model, diffusion, data):
    torch.multiprocessing.set_sharing_strategy('file_system')

    bs = args.batch_size
    args.num_classes = 40
    args.nfeats = 6
    args.njoint = 25
    device = dist_util.dev()


    recogparameters = args.__dict__.copy()
    recogparameters["pose_rep"] = "rot6d"
    recogparameters["nfeats"] = 6

    # Action2motionEvaluation
    stgcnevaluation = STGCNEvaluation(args.dataset, recogparameters, device)

    stgcn_metrics = {}

    data_types = ['train', 'test']
    datasetGT = {'train': [data], 'test': [copy.deepcopy(data)]}

    for key in data_types:
        datasetGT[key][0].split = key

    compute_gt_gt = False
    if compute_gt_gt:
        for key in data_types:
            datasetGT[key].append(copy.deepcopy(datasetGT[key][0]))

    model.eval()

    allseeds = list(range(args.num_seeds))

    for index, seed in enumerate(allseeds):
        print(f"Evaluation number: {index + 1}/{args.num_seeds}")
        fixseed(seed)
        for key in data_types:
            for data in datasetGT[key]:
                data.reset_shuffle()
                data.shuffle()

        dataiterator = {key: [DataLoader(data, batch_size=bs, shuffle=False, num_workers=8, collate_fn=collate)
                              for data in datasetGT[key]]
                        for key in data_types}

        new_data_loader = functools.partial(NewDataloader, model=model, diffusion=diffusion, device=device,
                                            cond_mode=args.cond_mode, dataset=args.dataset, num_samples=args.num_samples)
        gtLoaders = {key: new_data_loader(mode="gt", dataiterator=dataiterator[key][0])
                     for key in ["train", "test"]}

        if compute_gt_gt:
            gtLoaders2 = {key: new_data_loader(mode="gt", dataiterator=dataiterator[key][0])
                          for key in ["train", "test"]}

        genLoaders = {key: new_data_loader(mode="gen", dataiterator=dataiterator[key][0])
                      for key in ["train", "test"]}

        loaders = {"gen": genLoaders,
                   "gt": gtLoaders}

        if compute_gt_gt:
            loaders["gt2"] = gtLoaders2

        stgcn_metrics[seed] = stgcnevaluation.evaluate(model, loaders)
        del loaders

    metrics = {"feats": {key: [format_metrics(stgcn_metrics[seed])[key] for seed in allseeds] for key in stgcn_metrics[allseeds[0]]}}

    return metrics
