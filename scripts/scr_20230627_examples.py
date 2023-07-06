import sys as _sys
import torch

from sample import generate, edit
from visualize import render_mesh
from train import train_mdm
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from utils import dist_util
from data_loaders.get_data import get_dataset_loader

def text_to_motion():
    _sys.argv.append('--model_path')
    _sys.argv.append(r'X:\Git\motion-diffusion-model/save/humanml_trans_enc_512/model000200000.pt')
    _sys.argv.append('--num_samples')
    _sys.argv.append('3')
    _sys.argv.append('--num_repetitions')
    _sys.argv.append('3')
    _sys.argv.append('--text_prompt')
    _sys.argv.append("the person walked forward and is picking up his toolbox.")
    generate.main()

def action_to_motion():
    _sys.argv.append('--model_path')
    _sys.argv.append(r'X:\Git\motion-diffusion-model/save/humanact12/model000350000.pt')
    _sys.argv.append('--num_samples')
    _sys.argv.append('3')
    _sys.argv.append('--num_repetitions')
    _sys.argv.append('3')
    _sys.argv.append('--action_name')
    _sys.argv.append("drink")
    generate.main()

def unconstrained_motion():
    _sys.argv.append('--model_path')
    _sys.argv.append(r'X:\Git\motion-diffusion-model/save/unconstrained/model000450000.pt')
    _sys.argv.append('--num_samples')
    _sys.argv.append('3')
    _sys.argv.append('--num_repetitions')
    _sys.argv.append('3')
    generate.main()

def render():
    _sys.argv.append('--input_path')
    _sys.argv.append(r'X:\Git\motion-diffusion-model\save\humanml_trans_enc_512\samples_humanml_trans_enc_512_000200000_seed10_the_person_kneeled_down\sample00_rep00.mp4')
    render_mesh.main()

def edit_in_between():
    _sys.argv.append('--model_path')
    _sys.argv.append(r'X:\Git\motion-diffusion-model/save/humanml_trans_enc_512/model000200000.pt')
    #_sys.argv.append(r'X:\Git\motion-diffusion-model/save/unconstrained/model000450000.pt')
    _sys.argv.append('--edit_mode')
    _sys.argv.append('in_between')
    edit.main()

def training():
    _sys.argv.append('--save_dir')
    _sys.argv.append(r'X:\Git\motion-diffusion-model\save_model')
    _sys.argv.append('--dataset')
    _sys.argv.append('humanact12')
    _sys.argv.append('--cond_mask_prob')
    _sys.argv.append('0')
    _sys.argv.append('--lambda_rcxyz')
    _sys.argv.append('1')
    _sys.argv.append('--lambda_vel')
    _sys.argv.append('1')
    _sys.argv.append('--lambda_fc')
    _sys.argv.append('1')
    _sys.argv.append('--unconstrained')
    _sys.argv.append('--train_platform_type')
    _sys.argv.append('TensorboardPlatform')
    _sys.argv.append('--overwrite')

    train_mdm.main()

def save_to_onnx():

    _sys.argv.append('--model_path')
    #_sys.argv.append(r'X:\Git\motion-diffusion-model\save\humanml_trans_enc_512\model000200000.pt')
    _sys.argv.append(r'X:\Git\motion-diffusion-model\save\unconstrained/model000450000.pt')
    _sys.argv.append('--eval_mode')
    _sys.argv.append('full')

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

    data_loader = get_dataset_loader(name=args.dataset, num_frames=60, batch_size=args.batch_size,)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data_loader)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    onnx_args = []
    onnx_args.append(torch.randn([1, 25, 6, 60]))
    onnx_args.append(torch.randint(1,40,[1]))
    onnx_args = tuple(onnx_args)

    torch.onnx.export(model=model,
                      args=onnx_args,
                      f=args.model_path.replace('.pt','.onnx'),
                      input_names={'x', 'timesteps'},
                      opset_version=12)

if __name__ == "__main__":
    edit_in_between()