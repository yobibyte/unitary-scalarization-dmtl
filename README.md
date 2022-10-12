# In Defense of the Unitary Scalarization for Deep Multi-Task Learning

With [In Defense of the Unitary Scalarization for Deep Multi-Task Learning](https://arxiv.org/pdf/2201.04122), we show that a basic multi-task learning optimizer performs on par with specialized algorithms and suggest a possible explanation based on regularization.
This repository contains all the code necessary to replicate the findings described in the paper.

### TL;DR
Our advice to practitioners is the following:
_before adapting multi-task optimizers to the use-case at hand, or designing a new one, test whether optimizing for the
sum of the losses, along with standard regularization and stabilization techniques from the literature
(e.g., early stopping, weight decay, dropout), attains the target performance._

### BibTeX
If you use our code in your research, please cite:
```
@inproceedings{Kurin2022,
    title={In Defense of the Unitary Scalarization for Deep Multi-Task Learning},
    author={Kurin, Vitaly and De Palma, Alessandro and Kostrikov, Ilya and Whiteson, Shimon and Kumar, M. Pawan},
    booktitle={Neural Information Processing Systems},
    year={2022}
}
```

### Available Multi-Task Optimizers

The code provides implementations for the following multi-task optimizers:
- Unitary Scalarization (`Baseline`, optimizing the sum of the losses): our **recommended** optimizer
(*to be possibly paired with single-task regularization/stabilization techniques*);
- [PCGrad](http://arxiv.org/abs/2001.06782) (`PCGrad`);
- [MGDA](http://arxiv.org/abs/1810.04650) (`MGDA`);
- [IMTL](https://openreview.net/forum?id=IMPnRXEWpvr) (`IMTL`)
- [GradDrop](https://arxiv.org/abs/2010.06808) (`GradDrop`)
- [RLW](http://arxiv.org/abs/2111.10603) (`RLW`).

The optimizers are implemented under a unified interface, defined by the `MTLOptimizer` class in `optimizers/utils.py`.
The class is initialized from a PyTorch optimizer (and possibly method-dependent arguments). This optimizer will be used
to step on the "modified" gradient defined by the chosen multi-task optimizer.  
The `MTLOptimizer` class exposes the `iterate` method
which, given a list of per-task losses and possibly the shared representation (for encoder-decoder architectures), updates
network parameters in-place according to the chosen multi-task optimizer.
For usage examples see `supervised_experiments/train_multi_task.py`.

All optimizers can be coupled with standard regularization and stabilization techniques by relying on the relative
standard PyTorch implementations. For instance, l2 regularization can be used by passing a PyTorch optimizer with
non-null `weight_decay` to the initializer of the chosen `MTLOptimizer` child class.

## Supervised Learning Experiments

### Code setup

The experiments are performed in a Docker container, installed by running
`./supervised_experiments/build_supervised_docker.sh <device>`,
where `<device>` can be `cpu` or `cu101`. It will use CUDA by default.

### Datasets
The supervised experiments assume that the CelebA and Cityscapes dataset have been pre-downloaded
and stored in the `$DATA_FOLDER/celeba/` and `$DATA_FOLDER/cityscapes/` folders, respectively.
Please set the `DATA_FOLDER` variable to point to the appropriate directory.
No additional setup is needed for MultiMNIST, which is automatically downloaded in the project directory.

The datasets can be downloaded from:
- Cityscapes: [here](https://www.dropbox.com/sh/gaw6vh6qusoyms6/AADwWi0Tp3E3M4B2xzeGlsEna?dl=0) (6.1G), pre-processed by [mtan](https://github.com/lorenmt/mtan)
- CelebA: [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (23G)

### Running the code
The `scripts` folder contains script to reproduce each of the SL experiments.
The experiments of section 4.1 can be reproduced by using `scripts/mnist.sh`, `scripts/celeba.sh`, and `scripts/cityscapes.sh`.
Those in section 5 can be replicated via `scripts/regularization_celeba.sh`.
```
cd unitary-scalarization-dmtl
# Set DATA_FOLDER to point to the directory containing celeba/ and cityscapes/ (see Datasets above)
./supervised_docker_run.sh <device_id> # either GPU ID or cpu

# Scripts: scripts/celeba.sh, scripts/cityscapes.sh, scripts/mnist.sh, scripts/regularization_celeba.sh, scripts/signagnostic_graddrop_celeba.sh
./scripts/mnist.sh  # Run chosen supervised learning script
```

### Results

The code logs results to `wandb`.
Such logging can be disabled by appending the `--debug` flag to all the lines of the scripts mentioned above.
In any case, results are also saved locally in a pickled dictionary in `$DATA_FOLDER/saved_results/`.

## Reinforcement Learning Experiments

### Code setup

The experiments are performed in a Docker container, installed by running `./docker_build.sh <device>`,
where `<device>` can be `cpu` or `cu101`. It will use CUDA by default.

You need to have MuJoCo's activation key [mjkey.txt](https://www.roboti.us/file/mjkey.txt) in the rl_experiments folder in order for it to work.

### Benchmarks

The experiments use MT10 and MT50 benchmarks from [Metaworld](https://github.com/rlworkgroup/metaworld).
If the provided Dockerfile is used, they are installed by default and none additional setup is needed.

### Running the code

```
cd unitary-scalarization-dmtl             
./rl_docker_run.sh <device_id> # either GPU ID or cpu

# copy any script from configs_mt10, configs_mt50, configs_ablations as needed to the `mtrl` directory and run it
# example:
bash run_mtsac_mt10_buffer_rewnorm_4x_buf_actor_reg_3em4.sh 42 # the number is a random seed to use
```

## Acknowledgements

We would like to thank the authors of the following repositories, upon which we built the present codebase:  
[mtan](https://github.com/lorenmt/mtan), [MGDA](https://github.com/isl-org/MultiObjectiveOptimization),
[Pytorch-PCGrad](https://github.com/WeiChengTseng/Pytorch-PCGrad), [RLW](https://openreview.net/attachment?id=OdnNBNIdFul&name=supplementary_material),
[CARE](https://github.com/facebookresearch/mtrl).
