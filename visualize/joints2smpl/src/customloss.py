import torch
import torch.nn.functional as F
from visualize.joints2smpl.src import config

# Guassian
def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

# angle prior
def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(
        pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def body_fitting_loss(body_pose, betas, model_joints, camera_t, camera_center,
                      joints_2d, joints_conf, pose_prior,
                      focal_length=5000, sigma=100, pose_prior_weight=4.78,
                      shape_prior_weight=5, angle_prior_weight=15.2,
                      output='sum'):
    """
    Loss function for body fitting
    """
    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)

    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss


# --- get camera fitting loss -----
def camera_fitting_loss(model_joints, camera_t, camera_t_est, camera_center, 
                        joints_2d, joints_conf,
                        focal_length=5000, depth_loss_weight=100):
    """
    Loss function for camera optimization.
    """
    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    # get the indexed four
    op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    op_joints_ind = [config.JOINT_MAP[joint] for joint in op_joints]
    gt_joints = ['RHip', 'LHip', 'RShoulder', 'LShoulder']
    gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]

    reprojection_error_op = (joints_2d[:, op_joints_ind] -
                             projected_joints[:, op_joints_ind]) ** 2
    reprojection_error_gt = (joints_2d[:, gt_joints_ind] -
                             projected_joints[:, gt_joints_ind]) ** 2

    # Check if for each example in the batch all 4 OpenPose detections are valid, otherwise use the GT detections
    # OpenPose joints are more reliable for this task, so we prefer to use them if possible
    is_valid = (joints_conf[:, op_joints_ind].min(dim=-1)[0][:, None, None] > 0).float()
    reprojection_loss = (is_valid * reprojection_error_op + (1 - is_valid) * reprojection_error_gt).sum(dim=(1, 2))

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    total_loss = reprojection_loss + depth_loss
    return total_loss.sum()



 # #####--- body fitiing loss -----
def body_fitting_loss_3d(body_pose, preserve_pose,
                         betas, model_joints, camera_translation,
                         j3d, pose_prior,
                         joints3d_conf,
                         sigma=100, pose_prior_weight=4.78*1.5,
                         shape_prior_weight=5.0, angle_prior_weight=15.2,
                         joint_loss_weight=500.0,
                         pose_preserve_weight=0.0,
                         use_collision=False,
                         model_vertices=None, model_faces=None,
                         search_tree=None,  pen_distance=None,  filter_faces=None,
                         collision_loss_weight=1000
                         ):
    """
    Loss function for body fitting
    """
    batch_size = body_pose.shape[0]

    #joint3d_loss = (joint_loss_weight ** 2) * gmof((model_joints + camera_translation) - j3d, sigma).sum(dim=-1)
    
    joint3d_error = gmof((model_joints + camera_translation) - j3d, sigma)
    
    joint3d_loss_part = (joints3d_conf ** 2) * joint3d_error.sum(dim=-1)
    joint3d_loss = ((joint_loss_weight ** 2) * joint3d_loss_part).sum(dim=-1)
    
    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)
    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)
    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    collision_loss = 0.0
    # Calculate the loss due to interpenetration
    if use_collision:
        triangles = torch.index_select(
            model_vertices, 1,
            model_faces).view(batch_size, -1, 3, 3)

        with torch.no_grad():
            collision_idxs = search_tree(triangles)

        # Remove unwanted collisions
        if filter_faces is not None:
            collision_idxs = filter_faces(collision_idxs)

        if collision_idxs.ge(0).sum().item() > 0:
            collision_loss = torch.sum(collision_loss_weight * pen_distance(triangles, collision_idxs))
    
    pose_preserve_loss = (pose_preserve_weight ** 2) * ((body_pose - preserve_pose) ** 2).sum(dim=-1)

    # print('joint3d_loss', joint3d_loss.shape)
    # print('pose_prior_loss', pose_prior_loss.shape)
    # print('angle_prior_loss', angle_prior_loss.shape)
    # print('shape_prior_loss', shape_prior_loss.shape)
    # print('collision_loss', collision_loss)
    # print('pose_preserve_loss', pose_preserve_loss.shape)

    total_loss = joint3d_loss + pose_prior_loss + angle_prior_loss + shape_prior_loss + collision_loss + pose_preserve_loss

    return total_loss.sum()


# #####--- get camera fitting loss -----
def camera_fitting_loss_3d(model_joints, camera_t, camera_t_est,
                           j3d, joints_category="orig", depth_loss_weight=100.0):
    """
    Loss function for camera optimization.
    """
    model_joints = model_joints + camera_t
    # # get the indexed four
    # op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    # op_joints_ind = [config.JOINT_MAP[joint] for joint in op_joints]
    #
    # j3d_error_loss = (j3d[:, op_joints_ind] -
    #                          model_joints[:, op_joints_ind]) ** 2

    gt_joints = ['RHip', 'LHip', 'RShoulder', 'LShoulder']
    gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]
    
    if joints_category=="orig":
        select_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category=="AMASS":
        select_joints_ind = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints]
    else:
        print("NO SUCH JOINTS CATEGORY!")

    j3d_error_loss = (j3d[:, select_joints_ind] -
                      model_joints[:, gt_joints_ind]) ** 2

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight**2) *  (camera_t - camera_t_est)**2

    total_loss = j3d_error_loss +  depth_loss
    return total_loss.sum()
