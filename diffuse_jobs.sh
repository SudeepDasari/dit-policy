nice -n 19 python finetune.py exp_name=t3_base3 agent=diffusion agent.features.restore_path=/home/sudeep/dp_pt/IN_1M.pth buffer_path=/scratch/sudeep/toaster3/buf.pkl wandb.debug=False task.train_buffer.cam_idx=2 max_iterations=100000  trainer=bc_step_sched
nice -n 19 python finetune.py exp_name=t3_base3 agent=diffusion agent.features.restore_path=/home/sudeep/dp_pt/IN_1M.pth buffer_path=/scratch/sudeep/toaster3/buf.pkl wandb.debug=False task.train_buffer.cam_idx=2 max_iterations=100000  trainer=bc_step_sched agent.noise_net_kwargs.hidden_dim=512

nice -n 19 python finetune.py exp_name=t3_base2 agent=diffusion agent.features.restore_path=/home/sudeep/dp_pt/IN_1M.pth buffer_path=/scratch/sudeep/toaster3/buf.pkl wandb.debug=False task.train_buffer.cam_idx=2 trainer.lr=0.0001 trainer.weight_decay=0.0001;
nice -n 19 python finetune.py exp_name=t3_base2 agent=diffusion agent.features.restore_path=/home/sudeep/dp_pt/IN_1M.pth buffer_path=/scratch/sudeep/toaster3/buf.pkl wandb.debug=False task.train_buffer.cam_idx=2 trainer.lr=0.0005;
nice -n 19 python finetune.py exp_name=t3_base2 agent=diffusion agent.features.restore_path=/home/sudeep/dp_pt/IN_1M.pth buffer_path=/scratch/sudeep/toaster3/buf.pkl wandb.debug=False task.train_buffer.cam_idx=2 trainer.lr=0.001;
nice -n 19 python finetune.py exp_name=t3_base2 agent=diffusion agent.features.restore_path=/home/sudeep/dp_pt/IN_1M.pth buffer_path=/scratch/sudeep/toaster3/buf.pkl wandb.debug=False task.train_buffer.cam_idx=2 trainer.lr=0.0001 trainer.weight_decay=0;

# baselines 
# nice -n 19 python finetune.py exp_name=t3_base2 agent.features.restore_path=/home/sudeep/dp_pt/IN_1M.pth buffer_path=/scratch/sudeep/toaster3/buf.pkl wandb.debug=False task.train_buffer.cam_idx=2

# test
# python finetune.py exp_name=test agent=diffusion agent.features.restore_path=/home/sudeep/dp_pt/IN_1M.pth buffer_path=/scratch/sudeep/toaster3/buf.pkl wandb.debug=True task.train_buffer.cam_idx=2 batch_size=5


nice -n 19 python finetune.py exp_name=diffuse_toaster3 agent=diffusion_unet buffer_path=/scratch/sudeep/toaster3/diffusion/buf.pkl max_iterations=500000  trainer=bc_cos_sched ac_chunk=16 train_transform=medium task.train_buffer.cam_indexes=[0,2] img_chunk=2
nice -n 19 python finetune.py exp_name=diffuse_toaster3 agent=diffusion_unet buffer_path=/scratch/sudeep/toaster3/diffusion/buf.pkl max_iterations=500000  trainer=bc_cos_sched ac_chunk=16 train_transform=medium task.train_buffer.cam_indexes=[2] agent.features.feature_dim=256


# baseline command (ImageNet + GMM)
nice -n 19 python finetune.py exp_name=test wandb.name=vit_baseline buffer_path=/scratch/sudeep/toaster3/vel/buf.pkl max_iterations=50000  task.train_buffer.cam_indexes=[2] train_transform=hard agent.features.restore_path=/home/sudeep/dp_pt/IN_1M.pth

# w/ old net

nice -n 19 python finetune.py exp_name=diffuse_toaster3 agent=diffusion task=end_effector_r6 buffer_path=/scratch/sudeep/toaster3/diffusion/buf.pkl max_iterations=500000  trainer=bc_cos_sched ac_chunk=16 train_transform=medium task.train_buffer.cam_indexes=[2]


# old
# nice -n 19 python finetune.py exp_name=diffuse_toaster3 agent=diffusion_unet buffer_path=/scratch/sudeep/toaster3/abs/buf.pkl task.train_buffer.cam_idx=2 max_iterations=500000  trainer=bc_cos_sched ac_chunk=16 train_transform=diffusion
# nice -n 19 python finetune.py exp_name=test buffer_path=/scratch/sudeep/r2d2_octo/berkeley_pick_data/buf.pkl wandb.debug=True task.train_buffer.cam_idx=0 max_iterations=500000  trainer=bc_cos_sched ac_chunk=16 train_transform=diffusion batch_size=5