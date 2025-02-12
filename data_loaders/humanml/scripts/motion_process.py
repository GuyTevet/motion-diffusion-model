from os.path import join as pjoin

from data_loaders.humanml.common.skeleton import Skeleton
import numpy as np
import os
from data_loaders.humanml.common.quaternion import *
from data_loaders.humanml.utils.paramUtil import *

import torch
from tqdm import tqdm
from data_loaders.humanml_utils import HML_JOINT_NAMES, HML_EE_JOINT_NAMES

import random
from copy import copy, deepcopy

# positions (batch, joint_num, 3)
def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def extract_features(positions, feet_thre, n_raw_offsets, kinematic_chain, face_joint_indx, fid_r, fid_l):
    global_positions = positions.copy()
    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
        return feet_l, feet_r

    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(dataset.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data


def process_file(positions, feet_thre):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(dataset.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


# Recover global angle and positions for rotation dataset
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_root_rot_heading_ang(joints):
    
    '''Get Forward Direction'''
    face_joint_idx = [2, 1, 17, 16]
    # l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
    r_hip, l_hip, sdr_r, sdr_l = face_joint_idx  # Note the bugfix
    across1 = joints[:, r_hip] - joints[:, l_hip]
    across2 = joints[:, sdr_r] - joints[:, sdr_l]
    across = across1 + across2
    across = torch.nn.functional.normalize(across, dim=1)
    # print(across1.shape, across2.shape)

    # forward (batch_size, 3)
    forward = torch.cross(torch.tensor([[[0], [1], [0]]], dtype=across.dtype, device=across.device), across, axis=1)
    forward = torch.nn.functional.normalize(forward, dim=1)

    return torch.atan2(forward[:, 0], forward[:, 2])[:, None]

def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions

def recover_rot(data):
    # dataset [bs, seqlen, 263/251] HumanML/KIT
    joints_num = 22 if data.shape[-1] == 263 else 21
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_pos_pad = torch.cat([r_pos, torch.zeros_like(r_pos)], dim=-1).unsqueeze(-2)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)
    cont6d_params = torch.cat([cont6d_params, r_pos_pad], dim=-2)
    return cont6d_params


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions
'''
For Text2Motion Dataset
'''
'''
if __name__ == "__main__":
    example_id = "000021"
    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22
    # ds_num = 8
    data_dir = '../dataset/pose_data_raw/joints/'
    save_dir1 = '../dataset/pose_data_raw/new_joints/'
    save_dir2 = '../dataset/pose_data_raw/new_joint_vecs/'

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # Get offsets of target skeleton
    example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    # print(tgt_offsets)

    source_list = os.listdir(data_dir)
    frame_num = 0
    for source_file in tqdm(source_list):
        source_data = np.load(os.path.join(data_dir, source_file))[:, :joints_num]
        try:
            dataset, ground_positions, positions, l_velocity = process_file(source_data, 0.002)
            rec_ric_data = recover_from_ric(torch.from_numpy(dataset).unsqueeze(0).float(), joints_num)
            np.save(pjoin(save_dir1, source_file), rec_ric_data.squeeze().numpy())
            np.save(pjoin(save_dir2, source_file), dataset)
            frame_num += dataset.shape[0]
        except Exception as e:
            print(source_file)
            print(e)

    print('Total clips: %d, Frames: %d, Duration: %fm' %
          (len(source_list), frame_num, frame_num / 20 / 60))
'''

if __name__ == "__main__":
    example_id = "03950_gt"
    # Lower legs
    l_idx1, l_idx2 = 17, 18
    # Right/Left foot
    fid_r, fid_l = [14, 15], [19, 20]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [11, 16, 5, 8]
    # l_hip, r_hip
    r_hip, l_hip = 11, 16
    joints_num = 21
    # ds_num = 8
    data_dir = '../dataset/kit_mocap_dataset/joints/'
    save_dir1 = '../dataset/kit_mocap_dataset/new_joints/'
    save_dir2 = '../dataset/kit_mocap_dataset/new_joint_vecs/'

    n_raw_offsets = torch.from_numpy(kit_raw_offsets)
    kinematic_chain = kit_kinematic_chain

    '''Get offsets of target skeleton'''
    example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    # print(tgt_offsets)

    source_list = os.listdir(data_dir)
    frame_num = 0
    '''Read source dataset'''
    for source_file in tqdm(source_list):
        source_data = np.load(os.path.join(data_dir, source_file))[:, :joints_num]
        try:
            name = ''.join(source_file[:-7].split('_')) + '.npy'
            data, ground_positions, positions, l_velocity = process_file(source_data, 0.05)
            rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
            if np.isnan(rec_ric_data.numpy()).any():
                print(source_file)
                continue
            np.save(pjoin(save_dir1, name), rec_ric_data.squeeze().numpy())
            np.save(pjoin(save_dir2, name), data)
            frame_num += data.shape[0]
        except Exception as e:
            print(source_file)
            print(e)

    print('Total clips: %d, Frames: %d, Duration: %fm' %
          (len(source_list), frame_num, frame_num / 12.5 / 60))


def traj_global2vel(traj_positions, traj_rot):

    # traj_positions [bs, 2 (x,z), seqlen]
    # traj_positions [bs, 1 (z+, rad), seqlen]
    # return first 3 hml enries [bs, 3, seqlen-1]

    # skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    # # (seq_len, joints_num, 4)
    # quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

    bs, _, seqlen = traj_positions.shape
    traj_positions = traj_positions.permute(0, 2, 1)
    euler = torch.zeros([bs, 3, seqlen], dtype=traj_rot.dtype, device=traj_rot.device)
    euler[:, 1:2] = traj_rot
    euler = euler.permute(0, 2, 1).contiguous()
    traj_rot_quat = euler2quat(euler, 'yxz', deg=False)

    # '''Quaternion to continuous 6D'''
    # cont_6d_params = quaternion_to_cont6d_np(quat_params)
    # # (seq_len, 4)
    r_rot = traj_rot_quat.clone()
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = torch.zeros_like(euler[:, 1:, :])
    velocity[:, :, [0,2]] = (traj_positions[:, 1:, :] - traj_positions[:, :-1, :]).clone()
    #     print(r_rot.shape, velocity.shape)
    velocity = qrot(r_rot[:, 1:], velocity)
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = qmul(r_rot[:, 1:].contiguous(), qinv(r_rot[:, :-1]))
    # (seq_len, joints_num, 4)

    r_velocity = torch.arcsin(r_velocity[:, :, 2:3])
    l_velocity = velocity[:, :, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = torch.cat([r_velocity, l_velocity], axis=-1).permute(0, 2, 1)[:, :, None]

    return root_data

def get_target_location(motion, mean, std, lengths, joints_num, all_goal_joint_names, target_joint_names, is_heading):
    assert (lengths == lengths[0]).all(), 'currently supporting only fixed length'
    batch_size = motion.shape[0]
    extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']  # todo: fix hardcoded indexing that assumes traj and heading are last      
   
    # output tensor
    target_loc = torch.zeros((batch_size, len(extended_goal_joint_names), 3, lengths[0]), dtype=motion.dtype, device=motion.device)  #  n_samples x (n_target_joints+1) x 3 x n_frames

    # hml to abs loc (all joints, not only the requested ones)
    joints_loc = hml_to_abs_loc(motion, mean, std, joints_num)
    pelvis_loc = HML_JOINT_NAMES.index('pelvis')  
    joints_loc = torch.concat([joints_loc, joints_loc[:, pelvis_loc:pelvis_loc+1]], dim=1)  # concatenate the pelvis location to be used for traj 
    
    # joint names to indices
    HML_JOINT_NAMES_w_traj = HML_JOINT_NAMES + ['traj']
    for sample_idx in range(batch_size):
        req_joint_idx_in = [HML_JOINT_NAMES_w_traj.index(name) for name in target_joint_names[sample_idx]]
        req_joint_idx_out = [extended_goal_joint_names.index(name) for name in target_joint_names[sample_idx]]    
    
        target_loc[sample_idx, req_joint_idx_out] = joints_loc[sample_idx, req_joint_idx_in]  # assign joints loc to output tensor
    
    target_loc[:, -2, 1] = 0   # zero the y axis for the trajectory
        
    # last entry is the heading
    heading = recover_root_rot_heading_ang(joints_loc)
    target_loc[:, -1:, 0][is_heading] = heading[is_heading]
    
    return target_loc[..., -1]  # return last frame only


def hml_to_abs_loc(motion, mean, std, joints_num):
    # hml to abs loc (all joints, not only the requested ones)
    unnormed_motion = (motion * std + mean).permute(0, 2, 3, 1).float()
    joints_loc = recover_from_ric(unnormed_motion, joints_num)
    joints_loc = joints_loc.view(-1, *joints_loc.shape[2:]).permute(0, 2, 3, 1)  # n_samples x n_joints x 3 x n_frames
    return joints_loc


def sample_goal(batch_size, device, force_joints=None):
    if force_joints is None:
        choices = np.array(['None', 'traj', 'pelvis'] + HML_EE_JOINT_NAMES)  # todo: fix hardcoded 'pelvis' ('traj' is ok because it's our convention)  
        none_prob = 0.5  # todo: maybe convert to an argument
        probabilities = torch.ones(len(choices)) * (1-none_prob) / (len(choices)  -1)
        probabilities[0] = none_prob  # None's probability 
        assert probabilities.sum() - 1 < 1e-6, 'probabilities should sum to 1'
        max_goal_joints_per_sample = 2
        # target_cond_idx = torch.randint(low=0, high=len(choices), size=(batch_size,max_goal_joints_per_sample))
        target_cond_idx = torch.multinomial(probabilities, max_goal_joints_per_sample * batch_size, replacement=True).view(batch_size, max_goal_joints_per_sample)    
        names = choices[target_cond_idx]
        names = np.array([np.unique(name) for name in names])
        names = np.array([np.delete(name, np.argwhere(name=='None')) for name in names])
        is_heading = torch.bernoulli(torch.ones(batch_size, device=device) * .5).to(bool)
    else:
        options = get_allowed_joint_options(force_joints)
        names = [copy(random.choice(options)) for _ in range(batch_size)]
        is_heading = torch.zeros(batch_size, device=device).to(bool)
        for i, n in enumerate(names):
            if 'heading' in n:
                is_heading[i] = True
                del n[n.index('heading')]
    return names, is_heading

def get_allowed_joint_options(config_name):
    if config_name == 'DIMP_FULL':
        return [['pelvis', 'heading'], ['pelvis', 'head'], ['traj', 'heading'], ['right_wrist', 'heading'], ['left_wrist', 'heading'], ['right_foot', 'heading'], ['left_foot', 'heading']]
    elif config_name == 'DIMP_FINAL':
        return [['pelvis', 'heading'], ['traj', 'heading'], ['right_wrist', 'heading'], ['left_wrist', 'heading'], ['right_foot', 'heading'], ['left_foot', 'heading'], []]
    elif config_name == 'DIMP_SLIM':
        return [['pelvis', 'heading'], ['pelvis', 'head'], ['traj', 'heading'], ['left_wrist', 'heading'], ['left_foot', 'heading']]
    elif config_name == 'DIMP_BENCH':
        return [['pelvis', 'heading'], ['pelvis', 'head']]
    elif config_name == 'PURE_T2M':
        return [[]]
    else:
        return [config_name.split(',')]
    
