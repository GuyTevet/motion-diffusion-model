from model.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
import torch
from visualize.simplify_loc2rot import joints2smpl

class npy2obj:
    def __init__(self, npy_path, sample_idx, rep_idx, device=0, cuda=True):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True)
        if self.npy_path.endswith('.npz'):
            self.motions = self.motions['arr_0']
        self.motions = self.motions[None][0]
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.opt_cache = {}
        self.sample_idx = sample_idx
        self.total_num_samples = self.motions['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.rep_idx*self.total_num_samples + self.sample_idx
        self.num_frames = self.motions['motion'][self.absl_idx].shape[-1]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)

        if self.nfeats == 3:
            print(f'Running SMPLify For sample [{sample_idx}], repetition [{rep_idx}], it may take a few minutes.')
            motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
            self.motions['motion'] = motion_tensor.cpu().numpy()
        elif self.nfeats == 6:
            self.motions['motion'] = self.motions['motion'][[self.absl_idx]]
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.real_num_frames = self.motions['lengths'][self.absl_idx]

        self.vertices = self.rot2xyz(torch.tensor(self.motions['motion']), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True)
        self.root_loc = self.motions['motion'][:, -1, :3, :].reshape(1, 1, 3, -1)
        # self.vertices += self.root_loc

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def save_npy(self, save_path):
        data_dict = {
            'motion': self.motions['motion'][0, :, :, :self.real_num_frames],
            'thetas': self.motions['motion'][0, :-1, :, :self.real_num_frames],
            'root_translation': self.motions['motion'][0, -1, :3, :self.real_num_frames],
            'faces': self.faces,
            'vertices': self.vertices[0, :, :, :self.real_num_frames],
            'text': self.motions['text'][0],
            'length': self.real_num_frames,
        }
        np.save(save_path, data_dict)
        
        # Conversion of data for .pkl file for 'softcat477/SMPL-to-FBX/' 
        # Change 'FbxTime.eFrames60's in SMPL-to-FBX/FbxReadWriter.py to 'FbxTime.eFrames30'
        # Known issues: Glitches below 30 fps exports in 'softcat477/SMPL-to-FBX/' (20 fps expected)
        import utils.rotation_conversions as geometry
        i = range(self.real_num_frames)
        poses_raw = []
        latest = None
        for idx, x in np.ndenumerate(torch.flatten(geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(torch.tensor(self.motions['motion'][0, :-1, :, i]))))):
            poses_raw.append(x)
            latest = x
        smpl_poses = np.array(poses_raw).reshape(self.real_num_frames, 72)
        trans_raw = []
        latest = None
        for idx, x in np.ndenumerate(self.motions['motion'][0, -1, :3, i]):   
            trans_raw.append(x)
            latest = x
        smpl_trans = np.array(trans_raw).reshape(self.real_num_frames, 3)*np.array([100, 1, 100])     
        
        data_dict2 = {'smpl_poses': smpl_poses,'smpl_trans': smpl_trans,}
        import pickle
        with open(save_path +".pkl", 'wb') as pickle_file:
            pickle.dump(data_dict2, pickle_file)     
        # End of Conversion of data for .pkl file for 'softcat477/SMPL-to-FBX/'

