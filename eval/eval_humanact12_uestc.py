"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import torch
import re

from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from eval.a2m.tools import save_metrics
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip


def evaluate(args, model, diffusion, data):
    scale = None
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
        scale = {
            'action': torch.ones(args.batch_size) * args.guidance_param,
        }
    model.to(dist_util.dev())
    model.eval()  # disable random masking


    folder, ckpt_name = os.path.split(args.model_path)
    if args.dataset == "humanact12":
        from eval.a2m.gru_eval import evaluate
        eval_results = evaluate(args, model, diffusion, data)
    elif args.dataset == "uestc":
        from eval.a2m.stgcn_eval import evaluate
        eval_results = evaluate(args, model, diffusion, data)
    else:
        raise NotImplementedError("This dataset is not supported.")

    # save results
    iter = int(re.findall('\d+', ckpt_name)[0])
    scale = 1 if scale is None else scale['action'][0].item()
    scale = str(scale).replace('.', 'p')
    metricname = "evaluation_results_iter{}_samp{}_scale{}_a2m.yaml".format(iter, args.num_samples, scale)
    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, eval_results)

    return eval_results


def main():
    args = evaluation_parser()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    print(f'Eval mode [{args.eval_mode}]')
    assert args.eval_mode in ['debug', 'full'], f'eval_mode {args.eval_mode} is not supported for dataset {args.dataset}'
    if args.eval_mode == 'debug':
        args.num_samples = 10
        args.num_seeds = 2
    else:
        args.num_samples = 1000
        args.num_seeds = 20
    args.cond_mode = 'action' # temporary code till 'unconstrained' is implemented


    data_loader = get_dataset_loader(name=args.dataset, num_frames=60, batch_size=args.batch_size,)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data_loader)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    eval_results = evaluate(args, model, diffusion, data_loader.dataset)

    fid_to_print = {k : sum([float(vv) for vv in v])/len(v) for k, v in eval_results['feats'].items() if 'fid' in k and 'gen' in k}
    print(fid_to_print)

if __name__ == '__main__':
    main()
