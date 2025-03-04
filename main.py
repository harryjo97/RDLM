"""Training and evaluation"""

import hydra
import os
import numpy as np
import utils.utils as utils
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define useful resolver for hydra config
OmegaConf.register_new_resolver("int", lambda x: int(x), replace=True)
OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
OmegaConf.register_new_resolver("str", lambda x: str(x), replace=True)
OmegaConf.register_new_resolver("prod", lambda x: np.prod(x), replace=True)

# path2name and path2data required for creating directory in run_mode=sample
OmegaConf.register_new_resolver(
    "path2name", lambda x: x.split('/')[-4]+"_"+x.split('/')[-1].replace(".pth",""), 
    replace=True
)
OmegaConf.register_new_resolver(
    "path2data", lambda x: x.split('/')[-6], 
    replace=True
)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg):
    ngpus = cfg.ngpus
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir if hydra_cfg.mode == RunMode.RUN else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    
    utils.makedirs(work_dir)

    with open_dict(cfg):
        cfg.ngpus = ngpus
        cfg.work_dir = work_dir
        cfg.wandb_run_name = hydra_cfg.job.override_dirname

	# Run the training pipeline
    port = int(np.random.randint(10000, 20000))
    logger = utils.get_logger(os.path.join(work_dir, "run.log"))

    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode != RunMode.RUN:
        logger.info(f"Run id: {hydra_cfg.job.id}")

    try:
        mp.set_start_method("forkserver")

        if cfg.run_mode=="train":
            import run_train
            run_multiprocess = run_train.run_multiprocess
        elif cfg.run_mode=="sample":
            import run_sample
            run_multiprocess = run_sample.run_multiprocess
        else:
            raise NotImplementedError(f"Run mode: {cfg.run_mode} not implemented.")
        
        mp.spawn(run_multiprocess, args=(ngpus, cfg, port), nprocs=ngpus, join=True)

    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()