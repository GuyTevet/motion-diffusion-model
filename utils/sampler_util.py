import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from utils.misc import wrapped_getattr
import joblib

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode
        self.encode_text = self.model.encode_text

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))

    def __getattr__(self, name, default=None):
        # this method is reached only if name is not in self.__dict__.
        return wrapped_getattr(self, name, default=None)


class AutoRegressiveSampler():
    def __init__(self, args, sample_fn, required_frames=196):
        self.sample_fn = sample_fn
        self.args = args
        self.required_frames = required_frames
    
    def sample(self, model, shape, **kargs):
        bs = shape[0]
        n_iterations = (self.required_frames // self.args.pred_len) + int(self.required_frames % self.args.pred_len > 0)
        samples_buf = []
        cur_prefix = deepcopy(kargs['model_kwargs']['y']['prefix'])  # init with data
        dynamic_text_mode = type(kargs['model_kwargs']['y']['text'][0]) == list  # Text changes on the fly - prompt per prediction is provided as a list (instead of a single prompt)
        if self.args.autoregressive_include_prefix:
            samples_buf.append(cur_prefix)
        autoregressive_shape = list(deepcopy(shape))
        autoregressive_shape[-1] = self.args.pred_len
        
        # Autoregressive sampling
        for i in range(n_iterations):
            
            # Build the current kargs
            cur_kargs = deepcopy(kargs)
            cur_kargs['model_kwargs']['y']['prefix'] = cur_prefix
            if dynamic_text_mode:
                cur_kargs['model_kwargs']['y']['text'] = [s[i] for s in kargs['model_kwargs']['y']['text']]
                if model.text_encoder_type == 'bert':
                    cur_kargs['model_kwargs']['y']['text_embed'] = (cur_kargs['model_kwargs']['y']['text_embed'][0][:, :, i], cur_kargs['model_kwargs']['y']['text_embed'][1][:, i])
                else:
                    raise NotImplementedError('DiP model only supports BERT text encoder at the moment. If you implement this, please send a PR!')
            
            # Sample the next prediction
            sample = self.sample_fn(model, autoregressive_shape, **cur_kargs)

            # Buffer the sample
            samples_buf.append(sample.clone()[..., -self.args.pred_len:])

            # Update the prefix
            cur_prefix = sample.clone()[..., -self.args.context_len:]

        full_batch = torch.cat(samples_buf, dim=-1)[..., :self.required_frames]  # 200 -> 196
        return full_batch