import numpy as np

# Map joints Name to SMPL joints idx
JOINT_MAP = {
'MidHip': 0,
'LHip': 1, 'LKnee': 4, 'LAnkle': 7, 'LFoot': 10,
'RHip': 2, 'RKnee': 5, 'RAnkle': 8, 'RFoot': 11,
'LShoulder': 16, 'LElbow': 18, 'LWrist': 20, 'LHand': 22, 
'RShoulder': 17, 'RElbow': 19, 'RWrist': 21, 'RHand': 23,
'spine1': 3, 'spine2': 6, 'spine3': 9,  'Neck': 12, 'Head': 15,
'LCollar':13, 'Rcollar' :14, 
'Nose':24, 'REye':26,  'LEye':26,  'REar':27,  'LEar':28, 
'LHeel': 31, 'RHeel': 34,
'OP RShoulder': 17, 'OP LShoulder': 16,
'OP RHip': 2, 'OP LHip': 1,
'OP Neck': 12,
}

full_smpl_idx = range(24)
key_smpl_idx = [0, 1, 4, 7,  2, 5, 8,  17, 19, 21,  16, 18, 20]


AMASS_JOINT_MAP = {
'MidHip': 0,
'LHip': 1, 'LKnee': 4, 'LAnkle': 7, 'LFoot': 10,
'RHip': 2, 'RKnee': 5, 'RAnkle': 8, 'RFoot': 11,
'LShoulder': 16, 'LElbow': 18, 'LWrist': 20,  
'RShoulder': 17, 'RElbow': 19, 'RWrist': 21, 
'spine1': 3, 'spine2': 6, 'spine3': 9,  'Neck': 12, 'Head': 15,
'LCollar':13, 'Rcollar' :14, 
}
amass_idx =       range(22)
amass_smpl_idx =  range(22)


SMPL_MODEL_DIR = "./body_models/"
GMM_MODEL_DIR = "./visualize/joints2smpl/smpl_models/"
SMPL_MEAN_FILE = "./visualize/joints2smpl/smpl_models/neutral_smpl_mean_params.h5"
# for collsion 
Part_Seg_DIR = "./visualize/joints2smpl/smpl_models/smplx_parts_segm.pkl"