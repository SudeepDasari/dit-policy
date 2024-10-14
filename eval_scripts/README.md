# Eval Scripts

We provide some example evaluation scrips that will allow you to run our policies on ALOHA and DROID robots.
* `eval_aloha.py` will deploy our policies on an ALOHA robot assuming the default 14-DoF joint-state action space. It provides an implementation for both temporal ensembling and receding horizon control with chunked action predictions.
* `eval_droid.py` will deploy our policies on a DROID robot using the default cartesian velocity action space. Predicted actions are directly executed on the robot (no test time smoothing).
* `eval_droid_state.py` will deploy our policies on a DROID robot using the cartesian position action space. Note that the rotation actions are predicted using a R6 representation (conversion code [here](https://github.com/AGI-Labs/r2d2_to_robobuf/blob/main/converter.py)) following [Chi et. al.](https://diffusion-policy.cs.columbia.edu). After prediction the actions are further smoothed with temporal ensembling. This action space is ideal for diffusion policy (U-Net) action heads.


## Setup Instructions

Just download the policy folder (produced by `finetune.py`) and add a file named `obs_config.yaml` to it. This will tell the eval script how to process the observations for the policy. An example is provided below:

```
img: '26638268_left'
transform:
  _target_: data4robotics.transforms.get_transform_by_name
  name: preproc
```
