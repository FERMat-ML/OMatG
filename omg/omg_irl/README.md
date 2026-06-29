# Open Materials Generation with Inference-Time Reinforcement Learning (OMatG-IRL)

[![Static Badge](https://img.shields.io/badge/ICML_2026-OpenReview.net-811913?labelColor=222529)](https://openreview.net/forum?id=xfHppnGXaH)
[![arXiv](https://img.shields.io/badge/arXiv-2602.00424-maroon)](https://arxiv.org/abs/2602.00424)

A policy-gradient reinforcement-learning (RL) framework for aligning pretrained OMatG models with downstream
objectives using black-box reward functions. This part of the OMatG framework accompanies the OMatG-IRL
[ICML 2026 paper](https://openreview.net/forum?id=xfHppnGXaH), which should be
[cited](#citing-omatg-irl) when using it.

This README focuses on the parts specific to the RL setup of OMatG-IRL in the `omg_irl` package. The 
[main README](../../README.md) describes the
general OMatG framework including installation, datasets, training of the underlying generative models, and the
evaluation metrics.

## Table of Contents

- [Overview.](#overview)
- [Installation.](#installation)
- [Reward Functions.](#reward-functions)
- [Training.](#training)
- [Generation.](#generation)
- [Plotting Learned Velocity-Annealing Schedules.](#plotting-learned-velocity-annealing-schedules)
- [Citing OMatG-IRL.](#citing-omatg-irl)

## Overview

OMatG-IRL updates a pretrained OMatG model through group-relative policy optimization (GRPO) in conjunction with 
proximal policy optimization (PPO). It understands the numerical integration of OMatG during generation as a Markov
decision process with a stochastic policy that samples the next state $x_{t+\Delta t}$ of the system given the current 
state $x_t$. Every RL training step performs the following loop:

1. Sample $G$ initial structures $x_0$ from the base distribution $p_0$ under identical conditioning. For the
crystal structure prediction task, this corresponds to sampling $G$ initial structures for the same composition.
2. Rollout $G$ trajectories by numerically integrating the current stochastic policy from every $x_0$ to $x_1$ with an
Euler&ndash;Maruyama scheme (without gradients).
3. Compute rewards and GRPO advantages on the final structures $x_1$ (without gradients). 
4. For the fixed group of trajectories, optimize the PPO objective for a given number of gradient steps (PPO epochs).
In addition to the PPO objective, one can also include a KL-regularization term to prevent the updated policy from 
deviating too much from the original pretrained policy.

If the pretrained OMatG model learned the velocity field $b^\theta(t,x_t)$ and denoiser $z^\theta(t,x_t)$ of a 
generative stochastic differential equation (SDE), policy-gradient RL can be applied directly to the resulting 
stochastic sampler through **score-based OMatG-IRL**. 

In contrast, when only the velocity field $b^\theta(t,x_t)$ of an ordinary differential equation (ODE) was learned, 
**velocity-based OMatG-IRL** adds stochastic perturbations to the ODE dynamics to enable policy-gradient RL. 

Using the same stochastic-perturbation idea, **velocity-annealing OMatG-IRL** uses policy-gradient RL to learn a 
time-dependent velocity-annealing schedule that rescales the frozen velocity field of a pretrained OMatG model.

OMatG-IRL can currently only be applied to reinforce the stochastic policies for the generative processes of the 
fractional coordinates (`pos` field) and lattice vectors (`cell` field). Optionally, one can disable RL for either 
field in which case it is passively integrated using the frozen OMatG base model without any RL. If the discrete 
species are integrated by the base model in a de novo generation setup, they are also passively integrated using the 
frozen OMatG base model without any RL. 

The [`conf_examples`](conf_examples) directory contains example configuration files that were used in the 
paper:
1. The [`score_based_omatg_irl`](conf_examples/score_based_omatg_irl) directory contains RL and OMatG configuration 
   files for score-based OMatG-IRL from Section 5.1 (see blue curve in Fig. 3). The pretrained checkpoint and original 
   OMatG configuration file is available on 
   [Hugging Face](https://huggingface.co/OMatG/MP-20-CSP/tree/main/Trig-SDE-Gamma).
2. The [`velocity_based_omatg_irl`](conf_examples/velocity_based_omatg_irl) directory contains RL and OMatG 
   configuration files for velocity-based OMatG-IRL from Section 5.1 (see orange curve in Fig. 3). The pretrained 
   checkpoint and original OMatG configuration file is available on 
   [Hugging Face](https://huggingface.co/OMatG/MP-20-CSP/tree/main/Trig-SDE-Gamma). The same directory also contains a 
   robust RL configuration in [`rl_config_robust.yaml`](conf_examples/velocity_based_omatg_irl/rl_config_robust.yaml) 
   that was used across different datasets and pretrained OMatG models in Appendix J.

<details>
<summary><b>Expand this section for details on the different OMatG-IRL variants.</b></summary>

### OMatG-IRL Variants

Each of the three OMatG-IRL variants corresponds to a class in the 
[`omg_irl_lightning`](omg_irl_lightning) package. The following table summarizes the differences between the three 
variants, where $\xi$ denotes a Gaussian noise term and $\sigma(t)$ is a configurable time-dependent noise schedule:

| Approach | Class | What is Learned                                                                                                                                                                                                             | Euler&ndash;Maruyama Update                                                                                                        |
|----------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Score-based | [`OMGIRLScore`](omg_irl_lightning/omg_irl_score.py) | Velocity field $b^\theta(t,x_t)$ and denoiser $z^\theta(t,x_t)$ through updates to the pretrained OMatG model.                                                                                                              | $x_{t+\Delta t} = x_t + [b^\theta - \frac{\sigma(t)^2}{2 \gamma(t)} z^\theta] \Delta t + \sigma(t) \sqrt{\Delta t} \xi$            |
| Velocity-based | [`OMGIRLVelocity`](omg_irl_lightning/omg_irl_velocity.py) | Velocity field $b^\theta(t, x_t)$ through updates to the pretrained OMatG model.                                                                                                                                            | $x_{t+\Delta t} = x_t + b^\theta \,\Delta t + \sigma(t) \sqrt{\Delta t} \xi$                                                       |
| Velocity-annealing | [`OMGIRLScale`](omg_irl_lightning/omg_irl_scale.py) | Time-dependent velocity-annealing schedule $s^\theta(t)$ through updates to a multilayer perceptron in [`ScaleMLP`](omg_irl_lightning/scale_mlp.py); the pretrained velocity field $b^{\theta_\mathrm{ref}}$ remains frozen. | $x_{t+\Delta t} = x_t + [1 + s^\theta(t)] b^{\theta_\mathrm{ref}} \Delta t + \sigma(t) b^{\theta_\mathrm{ref}} \sqrt{\Delta t} \xi$ |

There is also the [`OMGIRLScaledVelocity`](omg_irl_lightning/omg_irl_scaled_velocity.py) class which is similar to 
`OMGIRLVelocity` but additionally considers a previously learned time-dependent velocity-annealing schedule 
$s^\theta(t)$ from `OMGIRLScale`. This class can thus be used to update the velocity field $b^\theta(t,x_t)$ through 
updates to the pretrained OMatG model while keeping a previously learned velocity-annealing schedule $s^\theta(t)$ in
a `ScaleMLP` frozen.

The available noise schedules $\sigma(t)$ in [```noise_schedules.py```](noise_schedules.py) are:

| Schedule | Formula                        |
|----------|--------------------------------|
| `ConstantNoiseSchedule` | $\sigma(t) = a$                |
| `SqrtNoiseSchedule` | $\sigma(t) = a \sqrt{(1-t)/t}$ |

The noise schedules are related to the $\epsilon(t)$ functions in the stochastic interpolants of the base OMatG models 
in [`omg/si.epsilon.py`](../si/epsilon.py) by $\sigma^2(t) = 2\epsilon(t)$. There is also a learnable noise schedule
implemented in `MLPNoiseSchedule` that starts from a constant noise schedule and learns a time-dependent noise schedule 
through RL updates to a multilayer perceptron.

</details>

<details>
<summary><b>Expand this section for tips on how to set up new configuration files.</b></summary>

### Configuration Files

OMatG-IRL uses two configuration files. First, the configuration file of the pretrained OMatG base model (as described 
in the [main README](../../README.md)) and, second, an additional configuration file for the RL parameters. The RL configuration 
file follows the same [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli) pattern as the OMatG configuration file with `model`, `optimizer`, and 
`trainer` sections. 

Note that there is no `data` section because the datamodule is taken over from the OMatG configuration file. The 
validation and prediction dataloaders use the batch size from the base configuration file. The batch size of the 
training dataloader, however, is internally overridden to `grpo_num_groups` from the RL configuration file. Since every
GRPO group is replicated to contain `grpo_group_size` trajectories, the effective batch size for the training dataloader 
is `grpo_num_groups * grpo_group_size`.

Note also that the number of integration timesteps of the rollouts is determined by the `integration_time_steps` 
parameter in the configuration file of the base model. We recommend to use a low number of integration time steps for 
policy-gradient RL (e.g., 50). We also recommend to remove any velocity annealing from the configuration file of the 
pretrained OMatG model.

#### Model

The `class_path` of the `model` section selects one of the OMatG-IRL variants that are 
implemented as classes in the [`omg_irl_lightning`](omg_irl_lightning) package. The `init_args` of the `model` section 
then contain the hyperparameters for the selected OMatG-IRL variant. For example, the following configuration was used 
for the velocity-based OMatG-IRL experiments in Section 5.1 in the paper:

```yaml
model:
  class_path: omg.omg_irl.OMGIRLVelocity  # Velocity-based OMatG-IRL.
  init_args:
    disable_fields:  # Do not reinforce the generative process of the `cell` field.
      - cell
    noise_schedules:  # Noise schedule for the stochastic perturbations in the Euler&ndash;Maruyama updates.
      pos:
        class_path: omg.omg_irl.SqrtNoiseSchedule
        init_args:
          noise_scale: 0.1668100537200059
    reference_noise_schedules:  # Noise schedule for the reference policy in the KL-regularization term.
      pos:
        class_path: omg.omg_irl.SqrtNoiseSchedule
        init_args:
          noise_scale: 0.1668100537200059
    grpo_group_size: 64  # Each GRPO group contains 64 trajectories.
    grpo_num_groups: 16  # Each training batch contains 16 GRPO groups
    grpo_share_x_0: false  # Do not share the same initial states x_0 across trajectories in the same GRPO group.
    ppo_clip_epsilon: 0.2811556854743765  # PPO clip epsilon hyperparameter.
    ppo_epochs: 4  # Number of PPO epochs per GRPO iteration.
    position_normalization: per_atom_surrogate  # Normalize the structure-level terms in the PPO objective and KL regularization to remove bias towards larger structures.
    relative_costs:  # Relative weights for the different terms in the PPO policy objective and KL regularization.
      pos_policy: 2.2978380697905694
      pos_regularization: 0.001
    normalize_relative_costs: false  # Do not normalize the relative costs to sum to 1.
    reward:  # Reward function for the RL training that minimizes the energy of the generated structures.
      class_path: omg.omg_irl.EnergyReward
      init_args:
        device: cuda  # Compute energies with MACE on the GPU.
        default_dtype: float64
        invalid_penalty: 3.0
        polar_sine_cutoff: 0.001
        scale: 1.0
        structure_check_cutoff: 0.5
        volume_check_cutoff: 0.1
    gradient_clip_algorithm: norm  # Gradient clipping algorithm.
    gradient_clip_val: 1.0
```

All arguments are further documented in the docstrings of the respective classes.

#### Optimizer and Trainer

The `optimizer` and `trainer` sections follow the same LightningCLI pattern as in the OMatG configuration file (see the
[main README](../../README.md)). For example:

```yaml
optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: 0.00011705655329569856
trainer:
  callbacks:
  - class_path: lightning.pytorch.callbacks.ModelCheckpoint  # Save model checkpoint with maximum reward.
    init_args:
      filename: best_val_reward_mean
      mode: max
      monitor: val_reward_mean
      save_top_k: 1
      save_weights_only: false
  - class_path: lightning.pytorch.callbacks.ModelCheckpoint  # Save model checkpoint during every validation step.
    init_args:
      every_n_epochs: 1
      monitor: val_reward_mean
      save_top_k: -1
      save_weights_only: false
  - class_path: lightning.pytorch.callbacks.TQDMProgressBar  # Use TQDM progress bar (instead of the default Lightning progress bar).
    init_args:
      leave: true
  enable_progress_bar: true
  inference_mode: false  # MACE always requires gradients, even during validation and testing, so do not use inference mode.
  log_every_n_steps: 1
  max_steps: 1200  # 300 batches with 4 PPO epochs each.
  num_sanity_val_steps: -1  # Run validation loop before training to get a baseline reward.
  precision: 32-true
  val_check_interval: 10  # Run validation every 10 training batches.
```

Since the RL training is implemented with 
[Lightning's manual optimization](https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html), one cannot
enable gradient accumulation or gradient clipping through the `trainer` section. Gradient clipping can instead be 
enabled in the `model` section by setting the `gradient_clip_algorithm` and `gradient_clip_val` hyperparameters (see the 
example above). Gradient accumulation is currently not supported.

</details>

## Installation

OMatG-IRL is part of the `omg` package. Installing the `omg` package as described in the
[main README](../../README.md#installation) also provides the `omg_irl` command used below.

## Reward Functions

Rewards are maximized during reinforcement. Every reward function implements the
[`Reward`](rewards/abstracts.py) abstract class whose `compute` method returns a reward for every generated structure
together with an information dictionary of additional per-structure quantities that are averaged and logged (with a
`val_` prefix during validation). The following rewards are available:

- [`EnergyReward`](rewards/energy_reward.py): Negative energy per atom predicted by the
  [MACE-MPA-0](https://github.com/ACEsuit/mace) foundation model. Energies are computed
  sequentially through [ASE](https://ase-lib.org) on CPU, or batched with
  [TorchSim](https://github.com/Radical-AI/torch-sim) on GPU. Structures that fail validity checks (based on volume,
  minimum interatomic distance, and polar sine of the lattice) are optionally assigned a penalty. 
  This reward is most useful for the crystal structure prediction task where all structures in a GRPO group 
  share the same composition.
- [`EnergyAboveHullReward`](rewards/energy_above_hull_reward.py): Negative energy above the convex hull based on
  MACE-MPA-0 energies and the [LeMat-GenBench](https://github.com/LeMaterial/lemat-genbench) reference convex hull. In 
  contrast to the raw energy per atom, the energy above the hull is comparable across compositions, which makes this 
  reward suitable for the de novo generation task where the compositions within a GRPO group vary.
- [`CRMSEReward`](rewards/crmse_reward.py): Structural similarity to the reference structures of the corresponding
  dataset based on PyMatGen's `StructureMatcher` (see the corrected root-mean-square error
  described in the [main README](../../README.md#crystal-structure-prediction-metrics)). Generated structures are
  compared to all reference structures with the same reduced composition, and non-matching structures are penalized
  with the site tolerance `stol`. This reward is only recommended for velocity-annealing OMatG-IRL.
- [`NonTriclinicReward`](rewards/symmetry_reward.py): Binary reward of one if the structure has a space group number
  greater than two (that is, the structure is not triclinic) as determined by [spglib](https://spglib.readthedocs.io),
  and zero otherwise.
- [`NonCentrosymmetricReward`](rewards/symmetry_reward.py): Binary reward of one if the structure is both
  non-triclinic and non-centrosymmetric (no inversion symmetry), targeting materials relevant for piezoelectricity,
  ferroelectricity, and nonlinear optics, and zero otherwise.
- [`CompositeRewards`](rewards/composite_rewards.py): Weighted sum of several reward functions. Rewards with zero
  weight are skipped during training but still computed during validation and prediction, which is useful for
  monitoring quantities that are not reinforced.

All parameters of the reward functions are documented in the docstrings of the respective classes.

## Training

Run the following command to reinforce a pretrained OMatG model:

```bash
omg_irl fit --config=<rl_configuration_file.yaml> omg --config=<configuration_file.yaml> --ckpt_path=<checkpoint_file.ckpt>
```

Every command-line argument after the `omg` keyword configures the pretrained OMatG base model and uses the same syntax 
as the `omg` command described in the [main README](../../README.md) (without any subcommand). Here,
`<configuration_file.yaml>` and `<checkpoint_file.ckpt>` are the configuration file and checkpoint of the pretrained 
OMatG base model. 

Every command-line argument before the `omg` keyword configures the RL setup itself. The
`<rl_configuration_file.yaml>` is the RL configuration file. 

If you want to include a Wandb logger with a name, add the `--trainer.logger=WandbLogger --trainer.logger.name=<name>`
argument before the `omg` keyword. In order to restart the RL training from an OMatG-IRL checkpoint, add the
`--ckpt_path=<rl_checkpoint_file.ckpt>` argument before the `omg` keyword. In order to seed the random number
generators, use `--seed_everything=<seed>`.

The structures generated during validation can be stored by setting
`--model.init_args.validation_xyz_filename=<xyz_file>` before the `omg` keyword (the filenames are suffixed with the 
current epoch and step).

## Generation

For generating new structures in an xyz file based on a reinforced model, run the following command:

```bash
omg_irl predict --config=<rl_configuration_file.yaml> --ckpt_path=<rl_checkpoint_file.ckpt> --model.generation_xyz_filename=<xyz_file> omg --config=<configuration_file.yaml> --ckpt_path=<checkpoint_file.ckpt>
```

As for the `omg predict` command in the [main README](../../README.md#generation), this command will generate one
epoch of structures, that is, the number of generated structures is equal to the number of structures in the
prediction dataset specified in the base configuration file. For an xyz filename `filename.xyz`, this command will
also create a file `filename_init.xyz` that contains the initial structures that were integrated to yield the
structures in `filename.xyz`.

## Plotting Learned Velocity-Annealing Schedules

For velocity-annealing OMatG-IRL models based on the [`OMGIRLScale`](omg_irl_lightning/omg_irl_scale.py) class, run the 
following command to plot the learned velocity-annealing schedule $s^\theta(t)$ of every reinforced data field as a 
function of time:

```bash
omg_irl plot_schedule --config=<rl_configuration_file.yaml> --ckpt_path=<rl_checkpoint_file.ckpt> --plot_filename=<plot_name.pdf> omg --config=<configuration_file.yaml> --ckpt_path=<checkpoint_file.ckpt>
```

## Citing OMatG-IRL

Please cite the following paper (in addition to the papers mentioned in the
[main README](../../README.md#citing-omatg)) when using OMatG-IRL in your work:

```bibtex
@inproceedings{
    hoellmer2026,
    title={Open Materials Generation with Inference-Time Reinforcement Learning},
    author={Philipp H{\"o}llmer and Stefano Martiniani},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=xfHppnGXaH},
    archivePrefix={arXiv},
    eprint={2602.00424},
    primaryClass={cs.LG},
}
```
