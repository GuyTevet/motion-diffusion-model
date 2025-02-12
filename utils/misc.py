import torch
import torch.nn as nn


class WeightedSum(nn.Module):
    def __init__(self, num_rows):
        super(WeightedSum, self).__init__()
        # Initialize learnable weights
        self.weights = nn.Parameter(torch.randn(num_rows))

    def forward(self, x):
        # Ensure weights are normalized (optional)
        normalized_weights = self.weights / self.weights.sum()  # torch.softmax(self.weights, dim=0)
        # Compute the weighted sum of the rows
        weighted_sum = torch.matmul(normalized_weights, x)
        return weighted_sum


def wrapped_getattr(self, name, default=None, wrapped_member_name='model'):
    ''' should be called from wrappers of model classes such as ClassifierFreeSampleModel'''

    if isinstance(self, torch.nn.Module):
        # for descendants of nn.Module, name may be in self.__dict__[_parameters/_buffers/_modules] 
        # so we activate nn.Module.__getattr__ first.
        # Otherwise, we might encounter an infinite loop
        try:
            attr = torch.nn.Module.__getattr__(self, name)
        except AttributeError:
            wrapped_member = torch.nn.Module.__getattr__(self, wrapped_member_name)
            attr = getattr(wrapped_member, name, default)
    else:
        # the easy case, where self is not derived from nn.Module
        wrapped_member = getattr(self, wrapped_member_name)
        attr = getattr(wrapped_member, name, default)
    return attr        


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(
            type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray


def cleanexit():
    import sys
    import os
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def freeze_joints(x, joints_to_freeze):
    # Freezes selected joint *rotations* as they appear in the first frame
    # x [bs, [root+n_joints], joint_dim(6), seqlen]
    frozen = x.detach().clone()
    frozen[:, joints_to_freeze, :, :] = frozen[:, joints_to_freeze, :, :1]
    return frozen
