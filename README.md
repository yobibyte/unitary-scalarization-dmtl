# In Defense of the Unitary Scalarization for Deep Multi-Task Learning

Source code for "In Defense of the Unitary Scalarization for Deep Multi-Task Learning".

## Supervised Experiments

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

```
cd rct
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
cd rct             
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