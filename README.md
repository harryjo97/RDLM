# Riemannian Diffusion Langauge Model

Official Code Repository for the paper [Continuous Diffusion Model for Language Modeling](https://arxiv.org/abs/2502.11564)

We provide an implementation for Riemannian Diffusion Language Model (RDLM) on language modeling tasks.


## Dependencies
Create an environment with Python 3.9, and Pytorch 2.3.1. Install requirements with the following command:

```
pip install -r requirements.txt
```

## Running Experiments

### 1. Configurations
The configurations are provided in the `config/` directory in YAML format. 
- To use new dataset, refer to `configs/exp`.
- To use new model architecture, refer to `configs/model`.
- To use new type of generative process, refer to `configs/sde`.

### 2. Preparations
Datasetes are automatically downloaded when running the training script.
- Data cache directory is set to `data/`. This can be modified via `data.cache_dir` in the config file.

- To add new dataset or modify dataset/tokenizer setting, please refer to the `data.py`.

### 3. Training

To run on Text8 dataset use the following command:
```sh
CUDA_VISIBLE_DEVICES=0 python main.py \
    ngpus=1 \
    training.accum=1 \
    exp=text8 \
    sde=mixture \
    sde.step_thr=0.35 \
    scheduler=geometric \
    scheduler.weight_type=step \
    scheduler.left=0.3 \
    scheduler.right=0.6
```
- `ngpus` is the number of GPUs used for training
- `training.accum` is the number of gradient accumulation steps. 

Modify these two hyperparameters to fit your hardware.


Similarly, to run on One Billion Words (LM1B) dataset use the following command:
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    ngpus=4 \
    training.accum=1 \
    exp=lm1b \
    tokens=3 \
    sde=mixture \
    sde.rho_scale=1.14 \
    sde.step_thr=0.38 \
    scheduler=geometric \
    scheduler.weight_type=step \
    scheduler.left=0.3 \
    scheduler.right=0.75
```



### 4. Generation and Evaluation

Run the following command to generate samples and evaluate:
```sh
CUDA_VISIBLE_DEVICES=0 python main.py \
    ngpus=1 \
    run_mode=sample \
    server=sample \
    exp=sample_lm1b \
    "model_path='PATH_TO_MODEL_CHECKPOINT'" \
    seed=0
```

## Pretrained checkpoints 
The checkpoints for the models trained on Text8 and LM1B datasets are available in this [Google Drive folder](https://drive.google.com/drive/folders/1aDTZtPIxAxQrkaRSahjuWbkxX1OYq9CC?usp=sharing).

- Download `checkpoint.pth` and pass the path to the downloaded file to `PATH_TO_MODEL_CHECKPOINT`. Use the command provided in the above section to generate and evaluate the samples.

- Additional files `sde.pkl` and `config.yaml` are provided for reproducibility and further analysis.


## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work.

```
@article{jo2025RDLM,
  author    = {Jaehyeong Jo and=
               Sung Ju Hwang},
  title     = {Continuous Diffusion Model for Language Modeling},
  journal   = {arXiv:2502.11564},
  year      = {2025},
  url       = {https://arxiv.org/abs/2502.11564}
}
```