# vc-1 training command (velocity action space, gaussian mlp policy)
nice -n 19 python finetune.py agent/policy=gaussian_constant exp_name=octo_baselines wandb.name=vc1_baseline buffer_path=/path/to/vel/buf.pkl max_iterations=50000  task.train_buffer.cam_indexes=[<target_cam_id>] train_transform=hard agent.features.restore_path=/path/to/vc1.pth

# r3m training command (velocity action space, gaussian mlp policy)
nice -n 19 python finetune.py agent/features=r3m agent/policy=gaussian_constant exp_name=octo_baselines wandb.name=r3m_baseline buffer_path=/path/to/vel/buf.pkl max_iterations=50000  task.train_buffer.cam_indexes=[<target_cam_id>] train_transform=medium agent.features.size=50

# single-cam diffusion (position + r6 rotation action space)
nice -n 19 python finetune.py agent=diffusion_unet exp_name=octo_baselines wandb.name=diffusion_singlecam buffer_path=/path/to/abs_r6/buf.pkl max_iterations=500000  trainer=bc_cos_sched ac_chunk=16 train_transform=medium task.train_buffer.cam_indexes=[<target_cam_id>] agent.features.feature_dim=256

# wrist-cam + 2-step obs diffusion (position + r6 rotation action space)
nice -n 19 python finetune.py agent=diffusion_unet exp_name=octo_baselines wandb.name=diffusion_multicam buffer_path=/path/to/abs_r6/buf.pkl max_iterations=500000  trainer=bc_cos_sched ac_chunk=16 train_transform=medium task.train_buffer.cam_indexes=[<front_cam_id>, <wrist_cam_id>] task.train_buffer.cam_indexes=[0,2] img_chunk=2