from nha.util.render import create_intrinsics_matrix
import torch
from nha.models.nha_optimizer import NHAOptimizer
from nha.util.general import dict_2_device
import numpy as np

# todo 1: change the following paths
# optimized avatar
ckpt = "/root/autodl-tmp/output/nha/result/metaface/08/4cam/1017/standard/lightning_logs/version_0/checkpoints/last.ckpt"
# head tracking
tracking_results = "/root/autodl-tmp/output/nha/tracking/metaface/08_4cam/1016/tracking_0/tracked_flame_params.npz"
# output path
output_path = "/root/autodl-tmp/mesh/metaface/nha/08_1018.obj"

avatar = NHAOptimizer.load_from_checkpoint(ckpt).eval().cuda()
tr = np.load(tracking_results)

def control_avatar():
    """
    Manipulate the following parameters to control the avatar.
    """

    # expression parameters
    e0 = 0
    e1 = 0
    e2 = 0
    e3 = 0
    e4 = 0

    # global rotation
    rot0 = 0
    rot1 = np.pi
    rot2 = 0

    # neck rotation
    neck0 = 0
    neck1 = 0
    neck2 = 0

    # jaw movement
    jaw = 0

    expr = torch.zeros(100, dtype=torch.float) # 存储表情参数
    pose = torch.zeros(15, dtype=torch.float) # 存储姿态参数
    expr[0] = e0; expr[1] = e1; expr[2] = e2; expr[3] = e3; expr[4] = e4
    pose[0] = rot0; pose[1] = rot1;  pose[2] = rot2; pose[3] = neck0;  pose[4] = neck1; pose[5] = neck2; pose[6] = jaw
    
    print("finish getting parameters")
    return expr, pose

def save_mesh(expr = torch.zeros(100, dtype=torch.float),
              pose = torch.zeros(15, dtype=torch.float),
              camera_idx = 0,
              image_size = (512, 512)):
    
    img_h, img_w = image_size # image size 
    track_h, track_w = tr['image_size'] # tracking image size
    
    # create camera intrinsics

    cam_intrinsics = tr["K"][camera_idx]
    
    # creating batch with inputs to avatar
    rest_joint_rots = avatar._flame.get_neutral_joint_rotations()
    default_pose = torch.cat((rest_joint_rots["global"], 
                              rest_joint_rots["neck"], 
                              rest_joint_rots["jaw"], 
                              rest_joint_rots["eyes"],
                              rest_joint_rots["eyes"]
                             ), dim=0).cpu()
    
    '''
    flame_shape: 人脸的形状参数
    flame_expr: 人脸的表情参数
    flame_pose: 人脸的姿态参数
    flame_trans: 人脸的位置转换参数
    cam_intrinsic: 相机的内参
    cam_extrinsic: 相机的外参
    rgb: 人脸的RGB图像
    '''
    batch = dict(
                flame_shape = torch.from_numpy(tr["shape"][None]).float(),
                flame_expr = expr[None],
                flame_pose = (pose+default_pose)[None],
                flame_trans = torch.from_numpy(tr["translations"][camera_idx][[0]]).float(),
                cam_intrinsic=cam_intrinsics[None],
                cam_extrinsic=torch.from_numpy(tr["RT"][camera_idx]).float()[None],
                rgb=torch.zeros(1,3,img_h,img_w),
                camera_idx=torch.tensor([camera_idx]),)    
    
    batch = dict_2_device(batch, avatar.device)
    avatar.get_mesh(batch, output_path)
    print("finish getting mesh")

expr, pose = control_avatar()
save_mesh(expr, pose, camera_idx=0)