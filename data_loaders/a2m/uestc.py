import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import utils.rotation_conversions as geometry
import torch

from .dataset import Dataset
# from torch.utils.data import Dataset

action2motion_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]


def get_z(cam_s, cam_pos, joints, img_size, flength):
    """
    Solves for the depth offset of the model to approx. orth with persp camera.
    """
    # Translate the model itself: Solve the best z that maps to orth_proj points
    joints_orth_target = (cam_s * (joints[:, :2] + cam_pos) + 1) * 0.5 * img_size
    height3d = np.linalg.norm(np.max(joints[:, :2], axis=0) - np.min(joints[:, :2], axis=0))
    height2d = np.linalg.norm(np.max(joints_orth_target, axis=0) - np.min(joints_orth_target, axis=0))
    tz = np.array(flength * (height3d / height2d))
    return float(tz)


def get_trans_from_vibe(vibe, index, use_z=True):
    alltrans = []
    for t in range(vibe["joints3d"][index].shape[0]):
        # Convert crop cam to orig cam
        # No need! Because `convert_crop_cam_to_orig_img` from demoutils of vibe
        # does this already for us :)
        # Its format is: [sx, sy, tx, ty]
        cam_orig = vibe["orig_cam"][index][t]
        x = cam_orig[2]
        y = cam_orig[3]
        if use_z:
            z = get_z(cam_s=cam_orig[0],  # TODO: There are two scales instead of 1.
                      cam_pos=cam_orig[2:4],
                      joints=vibe['joints3d'][index][t],
                      img_size=540,
                      flength=500)
            # z = 500 / (0.5 * 480 * cam_orig[0])
        else:
            z = 0
        trans = [x, y, z]
        alltrans.append(trans)
    alltrans = np.array(alltrans)
    return alltrans - alltrans[0]


class UESTC(Dataset):
    dataname = "uestc"

    def __init__(self, datapath="dataset/uestc", method_name="vibe", view="all", **kargs):

        self.datapath = datapath
        self.method_name = method_name
        self.view = view
        super().__init__(**kargs)

        # Load pre-computed #frames data
        with open(os.path.join(datapath, 'info', 'num_frames_min.txt'), 'r') as f:
            num_frames_video = np.asarray([int(s) for s in f.read().splitlines()])

        # Out of 118 subjects -> 51 training, 67 in test
        all_subjects = np.arange(1, 119)
        self._tr_subjects = [
            1, 2, 6, 12, 13, 16, 21, 24, 28, 29, 30, 31, 33, 35, 39, 41, 42, 45, 47, 50,
            52, 54, 55, 57, 59, 61, 63, 64, 67, 69, 70, 71, 73, 77, 81, 84, 86, 87, 88,
            90, 91, 93, 96, 99, 102, 103, 104, 107, 108, 112, 113]
        self._test_subjects = [s for s in all_subjects if s not in self._tr_subjects]

        # Load names of 25600 videos
        with open(os.path.join(datapath, 'info', 'names.txt'), 'r') as f:
            videos = f.read().splitlines()

        self._videos = videos

        if self.method_name == "vibe":
            vibe_data_path = os.path.join(datapath, "vibe_cache_refined.pkl")
            vibe_data = pkl.load(open(vibe_data_path, "rb"))

            self._pose = vibe_data["pose"]
            num_frames_method = [p.shape[0] for p in self._pose]
            globpath = os.path.join(datapath, "globtrans_usez.pkl")

            if os.path.exists(globpath):
                self._globtrans = pkl.load(open(globpath, "rb"))
            else:
                self._globtrans = []
                for index in tqdm(range(len(self._pose))):
                    self._globtrans.append(get_trans_from_vibe(vibe_data, index, use_z=True))
                pkl.dump(self._globtrans, open("globtrans_usez.pkl", "wb"))
            self._joints = vibe_data["joints3d"]
            self._jointsIx = action2motion_joints
        else:
            raise ValueError("This method name is not recognized.")

        num_frames_video = np.minimum(num_frames_video, num_frames_method)
        num_frames_video = num_frames_video.astype(int)
        self._num_frames_in_video = [x for x in num_frames_video]

        N = len(videos)
        self._actions = np.zeros(N, dtype=int)
        for ind in range(N):
            self._actions[ind] = self.parse_action(videos[ind])

        self._actions = [x for x in self._actions]

        total_num_actions = 40
        self.num_actions = total_num_actions
        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}
        self.num_classes = len(keep_actions)

        self._train = []
        self._test = []

        self.info_actions = []

        def get_rotation(view):
            theta = - view * np.pi/4
            axis = torch.tensor([0, 1, 0], dtype=torch.float)
            axisangle = theta*axis
            matrix = geometry.axis_angle_to_matrix(axisangle)
            return matrix

        # 0 is identity if needed
        rotations = {key: get_rotation(key) for key in [0, 1, 2, 3, 4, 5, 6, 7]}

        for index, video in enumerate(tqdm(videos, desc='Preparing UESTC data..')):
            act, view, subject, side = self._get_action_view_subject_side(video)
            self.info_actions.append({"action": act,
                                      "view": view,
                                      "subject": subject,
                                      "side": side})
            if self.view == "frontview":
                if side != 1:
                    continue
            # rotate to front view
            if side != 1:
                # don't take the view 8 in side 2
                if view == 8:
                    continue
                rotation = rotations[view]
                global_matrix = geometry.axis_angle_to_matrix(torch.from_numpy(self._pose[index][:, :3]))
                # rotate the global pose
                self._pose[index][:, :3] = geometry.matrix_to_axis_angle(rotation @ global_matrix).numpy()
                # rotate the joints
                self._joints[index] = self._joints[index] @ rotation.T.numpy()
                self._globtrans[index] = (self._globtrans[index] @ rotation.T.numpy())

            # add the global translation to the joints
            self._joints[index] = self._joints[index] + self._globtrans[index][:, None]

            if subject in self._tr_subjects:
                self._train.append(index)
            elif subject in self._test_subjects:
                self._test.append(index)
            else:
                raise ValueError("This subject doesn't belong to any set.")

            # if index > 200:
            #     break

        # Select only sequences which have a minimum number of frames
        if self.num_frames > 0:
            threshold = self.num_frames*3/4
        else:
            threshold = 0

        method_extracted_ix = np.where(num_frames_video >= threshold)[0].tolist()
        self._train = list(set(self._train) & set(method_extracted_ix))
        # keep the test set without modification
        self._test = list(set(self._test))

        action_classes_file = os.path.join(datapath, "info/action_classes.txt")
        with open(action_classes_file, 'r') as f:
            self._action_classes = np.array(f.read().splitlines())

        # with open(processd_path, 'wb') as file:
        #     pkl.dump(xxx, file)

    def _load_joints3D(self, ind, frame_ix):
        if len(self._joints[ind]) == 0:
            raise ValueError(
                f"Cannot load index {ind} in _load_joints3D function.")
        if self._jointsIx is not None:
            joints3D = self._joints[ind][frame_ix][:, self._jointsIx]
        else:
            joints3D = self._joints[ind][frame_ix]

        return joints3D

    def _load_rotvec(self, ind, frame_ix):
        # 72 dim smpl
        pose = self._pose[ind][frame_ix, :].reshape(-1, 24, 3)
        return pose

    def _get_action_view_subject_side(self, videopath):
        # TODO: Can be moved to tools.py
        spl = videopath.split('_')
        action = int(spl[0][1:])
        view = int(spl[1][1:])
        subject = int(spl[2][1:])
        side = int(spl[3][1:])
        return action, view, subject, side

    def _get_videopath(self, action, view, subject, side):
        # Unused function
        return 'a{:d}_d{:d}_p{:03d}_c{:d}_color.avi'.format(
            action, view, subject, side)

    def parse_action(self, path, return_int=True):
        # Override parent method
        info, _, _, _ = self._get_action_view_subject_side(path)
        if return_int:
            return int(info)
        else:
            return info


if __name__ == "__main__":
    dataset = UESTC()
