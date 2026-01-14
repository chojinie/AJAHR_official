import torch

def merging_feats(joint_batch):
    batch = {}

#     smpl_params = {'global_orient': None,
#             'body_pose': None,
#             'betas': None
#             }
#     has_smpl_params = {'global_orient': None,
#             'body_pose': None,
#             'betas': None
#             }
#     smpl_params_is_axis_angle = {'global_orient': None,
#             'body_pose': None,
#             'betas': None
#             }
    
#     batch['smpl_params'] = smpl_params
#     batch['has_smpl_params'] = has_smpl_params
#     batch['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle

    batch['imgname'] = joint_batch['tar']['imgname'] + joint_batch['nontar']['imgname']
    
    batch['img'] = torch.concat((joint_batch['tar']['img'], joint_batch['nontar']['img']), dim=0)
    batch['keypoints_2d'] = torch.concat((joint_batch['tar']['keypoints_2d'], joint_batch['nontar']['keypoints_2d']), dim=0)
#     batch['keypoints_3d'] = torch.concat((joint_batch['tar']['keypoints_3d'], joint_batch['nontar']['keypoints_3d']), dim=0)
   
#     batch['smpl_params']['global_orient'] = torch.concat((joint_batch['tar']['smpl_params']['global_orient'], joint_batch['nontar']['smpl_params']['global_orient']), dim=0)
#     batch['smpl_params']['body_pose'] = torch.concat((joint_batch['tar']['smpl_params']['body_pose'], joint_batch['nontar']['smpl_params']['body_pose']), dim=0)
#     batch['smpl_params']['betas'] = torch.concat((joint_batch['tar']['smpl_params']['betas'], joint_batch['nontar']['smpl_params']['betas']), dim=0)
    
#     batch['has_smpl_params']['global_orient'] = torch.concat((joint_batch['tar']['has_smpl_params']['global_orient'], joint_batch['nontar']['has_smpl_params']['global_orient']), dim=0)
#     batch['has_smpl_params']['body_pose'] = torch.concat((joint_batch['tar']['has_smpl_params']['body_pose'], joint_batch['nontar']['has_smpl_params']['body_pose']), dim=0)
#     batch['has_smpl_params']['betas'] = torch.concat((joint_batch['tar']['has_smpl_params']['betas'], joint_batch['nontar']['has_smpl_params']['betas']), dim=0)
    
#     batch['smpl_params_is_axis_angle']['global_orient'] = torch.concat((joint_batch['tar']['smpl_params_is_axis_angle']['global_orient'], joint_batch['nontar']['smpl_params_is_axis_angle']['global_orient']), dim=0)
#     batch['smpl_params_is_axis_angle']['body_pose'] = torch.concat((joint_batch['tar']['smpl_params_is_axis_angle']['body_pose'], joint_batch['nontar']['smpl_params_is_axis_angle']['body_pose']), dim=0)
#     batch['smpl_params_is_axis_angle']['betas'] = torch.concat((joint_batch['tar']['smpl_params_is_axis_angle']['betas'], joint_batch['nontar']['smpl_params_is_axis_angle']['betas']), dim=0)
    
    batch['dataset'] = joint_batch['tar']['dataset'] + joint_batch['nontar']['dataset']
    batch['label'] = torch.cat([joint_batch['tar']['label'], joint_batch['nontar']['label']], dim=0)
    
    return shuffle_batch(batch)


def only_amp(joint_batch):
    batch = {}

    smpl_params = {'global_orient': None,
            'body_pose': None,
            'betas': None
            }
    has_smpl_params = {'global_orient': None,
            'body_pose': None,
            'betas': None
            }
    smpl_params_is_axis_angle = {'global_orient': None,
            'body_pose': None,
            'betas': None
            }
    
    batch['smpl_params'] = smpl_params
    batch['has_smpl_params'] = has_smpl_params
    batch['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle

    batch['imgname'] = joint_batch['nontar']['imgname']
    
    batch['img'] = joint_batch['nontar']['img']
    batch['keypoints_2d'] = joint_batch['nontar']['keypoints_2d']
    batch['keypoints_3d'] = joint_batch['nontar']['keypoints_3d']
   
    batch['smpl_params']['global_orient'] = joint_batch['nontar']['smpl_params']['global_orient']
    batch['smpl_params']['body_pose'] = joint_batch['nontar']['smpl_params']['body_pose']
    batch['smpl_params']['betas'] = joint_batch['nontar']['smpl_params']['betas']
    
    batch['has_smpl_params']['global_orient'] = joint_batch['nontar']['has_smpl_params']['global_orient']
    batch['has_smpl_params']['body_pose'] = joint_batch['nontar']['has_smpl_params']['body_pose']
    batch['has_smpl_params']['betas'] = joint_batch['nontar']['has_smpl_params']['betas']
    
    batch['smpl_params_is_axis_angle']['global_orient'] = joint_batch['nontar']['smpl_params_is_axis_angle']['global_orient']
    batch['smpl_params_is_axis_angle']['body_pose'] = joint_batch['nontar']['smpl_params_is_axis_angle']['body_pose']
    batch['smpl_params_is_axis_angle']['betas'] = joint_batch['nontar']['smpl_params_is_axis_angle']['betas']
    
    batch['dataset'] = joint_batch['nontar']['dataset']
    batch['label'] = joint_batch['nontar']['label']
    
    return shuffle_batch(batch)


def shuffle_batch(batch):
    num_samples = batch['label'].shape[0]
    indices = torch.randperm(num_samples)
    
    batch['imgname'] = [batch['imgname'][i] for i in indices]
    batch['img'] = batch['img'][indices]
    batch['keypoints_2d'] = batch['keypoints_2d'][indices]
#     batch['keypoints_3d'] = batch['keypoints_3d'][indices]
    
#     batch['smpl_params']['global_orient'] = batch['smpl_params']['global_orient'][indices]
#     batch['smpl_params']['body_pose'] = batch['smpl_params']['body_pose'][indices]
#     batch['smpl_params']['betas'] = batch['smpl_params']['betas'][indices]
    
#     batch['has_smpl_params']['global_orient'] = batch['has_smpl_params']['global_orient'][indices]
#     batch['has_smpl_params']['body_pose'] = batch['has_smpl_params']['body_pose'][indices]
#     batch['has_smpl_params']['betas'] = batch['has_smpl_params']['betas'][indices]
    
#     batch['smpl_params_is_axis_angle']['global_orient'] = batch['smpl_params_is_axis_angle']['global_orient'][indices]
#     batch['smpl_params_is_axis_angle']['body_pose'] = batch['smpl_params_is_axis_angle']['body_pose'][indices]
#     batch['smpl_params_is_axis_angle']['betas'] = batch['smpl_params_is_axis_angle']['betas'][indices]
    
    batch['dataset'] = [batch['dataset'][i] for i in indices]
    batch['label'] = batch['label'][indices]
    
    return batch