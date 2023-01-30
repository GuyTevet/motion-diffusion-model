import numpy as np
import torch

from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_euler_angles
from visualize.simplify_loc2rot import joints2smpl

"""
Utility function to convert model output to a representation used by HumanIK skeletons in Maya and Motion Builder
by converting joint positions to joint rotations in degrees. Based on visualize.vis_utils.npy2obj
"""

# Mapping of SMPL joint index to HIK joint Name
JOINT_MAP = [
    'Hips',
    'LeftUpLeg',
    'RightUpLeg',
    'Spine',
    'LeftLeg',
    'RightLeg',
    'Spine1',
    'LeftFoot',
    'RightFoot',
    'Spine2',
    'LeftToeBase',
    'RightToeBase',
    'Neck',
    'LeftShoulder',
    'RightShoulder',
    'Head',
    'LeftArm',
    'RightArm',
    'LeftForeArm',
    'RightForeArm',
    'LeftHand',
    'RightHand'
]


def motions2hik(motions,  device=0, cuda=True):
    """
    Utility function to convert model output to a representation used by HumanIK skeletons in Maya and Motion Builder
    by converting joint positions to joint rotations in degrees. Based on visualize.vis_utils.npy2obj

    :param motions: numpy array containing MDM model output [num_reps, num_joints, num_params (xyz), num_frames
    :param device:
    :param cuda:

    :returns: JSON serializable dict to be used with the Replicate API implementation
    """

    nreps, njoints, nfeats, nframes = motions.shape
    j2s = joints2smpl(num_frames=nframes, device_id=device, cuda=cuda)

    thetas = []
    root_translation = []
    for rep_idx in range(nreps):
        rep_motions = motions[rep_idx].transpose(2, 0, 1)  # [nframes, njoints, 3]

        if nfeats == 3:
            print(f'Running SMPLify for repetition [{rep_idx + 1}] of {nreps}, it may take a few minutes.')
            motion_tensor, opt_dict = j2s.joint2smpl(rep_motions)  # [nframes, njoints, 3]
            motion = motion_tensor.cpu().numpy()

        elif nfeats == 6:
            motion = rep_motions
            thetas.append(rep_motions)

        # Convert 6D rotation representation to Euler angles
        thetas_6d = motion[0, :-1, :, :nframes].transpose(2, 0, 1)  # [nframes, njoints, 6]
        thetas_deg = []
        for frame, d6 in enumerate(thetas_6d):
            thetas_deg.append([_rotation_6d_to_euler(d6)])

        thetas.append([np.concatenate(thetas_deg, axis=0)])
        root_translation.append([motion[0, -1, :3, :nframes].transpose(1, 0)])  # [nframes, 3]

    thetas = np.concatenate(thetas, axis=0)[:nframes]
    root_translation = np.concatenate(root_translation, axis=0)[:nframes]

    data_dict = {
        'joint_map': JOINT_MAP,
        'thetas': thetas.tolist(),  # [nreps, nframes, njoints, 3 (deg)]
        'root_translation': root_translation.tolist(), # [nreps, nframes, 3 (xyz)]
    }

    return data_dict


def _rotation_6d_to_euler(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to euler angles
    using Gram--Schmidt orthogonalisation per Section B of [1].

    :param d6: numpy Array 6D rotation representation, of size (*, 6)
        :returns: JSON serializable dict to be used with the Replicate API implementation
    :returns: euler angles in degrees as a numpy array with shape (*, 3)
    """
    rot_mat = rotation_6d_to_matrix(torch.tensor(d6))
    rot_eul_rad = matrix_to_euler_angles(rot_mat, 'XYZ')
    eul_deg = torch.rad2deg(rot_eul_rad).numpy()

    return eul_deg

