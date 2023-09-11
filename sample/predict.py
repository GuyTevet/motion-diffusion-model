import os
import subprocess
from typing import Any, List, Optional
from argparse import Namespace

import torch
from cog import BasePredictor, Input, Path, BaseModel

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
from model.cfg_sampler import ClassifierFreeSampleModel
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from visualize.motions2hik import motions2hik
from sample.generate import construct_template_variables

"""
In case of matplot lib issues it may be needed to delete model/data_loaders/humanml/utils/plot_script.py" in lines 89~92 as
suggested in https://github.com/GuyTevet/motion-diffusion-model/issues/6
"""


class ModelOutput(BaseModel):
    json_file: Optional[Any]
    animation: Optional[List[Path]]


def get_args():
    args = Namespace()
    args.fps = 20
    args.model_path = './save/humanml_trans_enc_512/model000200000.pt'
    args.guidance_param = 2.5
    args.unconstrained = False
    args.dataset = 'humanml'

    args.cond_mask_prob = 1
    args.emb_trans_dec = False
    args.latent_dim = 512
    args.layers = 8
    args.arch = 'trans_enc'

    args.noise_schedule = 'cosine'
    args.sigma_small = True
    args.lambda_vel = 0.0
    args.lambda_rcxyz = 0.0
    args.lambda_fc   = 0.0
    return args


class Predictor(BasePredictor):
    def setup(self):
        subprocess.run(["mkdir", "/root/.cache/clip"])
        subprocess.run(["cp", "-r", "ViT-B-32.pt", "/root/.cache/clip"])

        self.args = get_args()
        self.num_frames = self.args.fps * 6
        print('Loading dataset...')

        # temporary data
        self.data = get_dataset_loader(name=self.args.dataset,
                                  batch_size=1,
                                  num_frames=196,
                                  split='test',
                                  hml_mode='text_only')

        self.data.fixed_length = float(self.num_frames)

        print("Creating model and diffusion...")
        self.model, self.diffusion = create_model_and_diffusion(self.args, self.data)

        print(f"Loading checkpoints from...")
        state_dict = torch.load(self.args.model_path, map_location='cpu')
        load_model_wo_clip(self.model, state_dict)

        if self.args.guidance_param != 1:
           self.model = ClassifierFreeSampleModel(self.model)   # wrapping model with the classifier-free sampler
        self.model.to(dist_util.dev())
        self.model.eval()  # disable random masking

    def predict(
            self,
            prompt: str = Input(default="the person walked forward and is picking up his toolbox."),
            num_repetitions: int = Input(default=3, description="How many"),
            output_format: str = Input(
                description='Choose the format of the output, either an animation or a json file of the animation data.\
                The json format is: {"thetas": [...], "root_translation": [...], "joint_map": [...]}, where "thetas" \
                is an [nframes x njoints x 3] array of joint rotations in degrees, "root_translation" is an [nframes x 3] \
                array of (X, Y, Z) positions of the root, and "joint_map" is a list mapping the SMPL joint index to the\
                corresponding HumanIK joint name',
                default="animation",
                choices=["animation", "json_file"],
            ),
    ) -> ModelOutput:
        args = self.args
        args.num_repetitions = int(num_repetitions)

        self.data = get_dataset_loader(name=self.args.dataset,
                                  batch_size=args.num_repetitions,
                                  num_frames=self.num_frames,
                                  split='test',
                                  hml_mode='text_only')

        collate_args = [{'inp': torch.zeros(self.num_frames), 'tokens': None, 'lengths': self.num_frames, 'text': str(prompt)}]
        _, model_kwargs = collate(collate_args)

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.num_repetitions, device=dist_util.dev()) * args.guidance_param

        sample_fn = self.diffusion.p_sample_loop
        sample = sample_fn(
            self.model,
            (args.num_repetitions, self.model.njoints, self.model.nfeats, self.num_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from HumanML3D vector representation
        if self.model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = self.data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if self.model.data_rep in ['xyz', 'hml_vec'] else self.model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.num_repetitions,
                                                                                                self.num_frames).bool()
        sample = self.model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

        all_motions = sample.cpu().numpy()

        if output_format == 'json_file':
            data_dict = motions2hik(all_motions)
            return ModelOutput(json_file=data_dict)

        caption = str(prompt)

        skeleton = paramUtil.t2m_kinematic_chain

        sample_print_template, row_print_template, all_print_template, \
            sample_file_template, row_file_template, all_file_template = construct_template_variables(
            args.unconstrained)

        rep_files = []
        replicate_fnames = []
        for rep_i in range(args.num_repetitions):
            motion = all_motions[rep_i].transpose(2, 0, 1)[:self.num_frames]
            save_file = sample_file_template.format(1, rep_i)
            print(sample_print_template.format(caption, 1, rep_i, save_file))
            plot_3d_motion(save_file, skeleton, motion, dataset=args.dataset, title=caption, fps=args.fps)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(save_file)

            replicate_fnames.append(Path(save_file))

        return ModelOutput(animation=replicate_fnames)
