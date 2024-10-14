# RoboSuite Evaluation

We provide a bare-bones implementation to reproduce our `robomimic` sim evaluation results. First, install the following `dm-control` versions of [robosuite](https://github.com/SudeepDasari/robosuite/tree/restore_dit) and [robomimic](https://github.com/SudeepDasari/robomimic), along with their associated dependencies. You will also have to [download](https://github.com/SudeepDasari/robomimic/blob/restore_dit/robomimic/scripts/download_datasets.py) the robomomic dataset (no camera obs required in download) into `/path/to/robomimic/downloads`. Then run:

```
python finetune.py exp_name=test agent=diffusion task=[robomimic_lift/can/square/toolhand] agent/features=resnet_gn agent.features.restore_path=/path/to/resnet18/IN_1M_resnet18.pth  trainer=bc_cos_sched ac_chunk=10 eval_freq=15000 batch_size=350
```
