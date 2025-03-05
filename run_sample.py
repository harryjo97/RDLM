import datetime
import os
import os.path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import pickle
from tqdm import tqdm
import gc
from hydra.utils import instantiate
from omegaconf import OmegaConf

import data
import sampling
import utils.utils as utils
from model.ema import ExponentialMovingAverage
from hypersphere import Hypersphere
from evaluation import Eval
import utils.seq_utils as sutils

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )

    # for clean terminal print
    import sys
    f = open(os.devnull, "w")
    if rank != 0:
        sys.stdout = f
        sys.stderr = f


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


def _run(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    if rank == 0:
        utils.makedirs(sample_dir)

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "run.log"))
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    mprint(f"SEED: {cfg.seed}")
    utils.set_seed(cfg.seed+dist.get_rank())
    
    
    # load saved state from checkpoint
    loaded_state = torch.load(cfg.model_path, map_location=device)
    train_cfg = loaded_state['config']

    # Update sampling config from training config
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "model", train_cfg.model)
    OmegaConf.update(cfg, "training", train_cfg.training)
    OmegaConf.update(cfg, "data", train_cfg.data)
    OmegaConf.update(cfg, "tokens", train_cfg.tokens)
    OmegaConf.update(cfg, "original_tokens", train_cfg.original_tokens)
    
    if cfg.sampling.batch_per_gpu == 0:
        batch_per_gpu = train_cfg.training.batch_size // (train_cfg.ngpus * train_cfg.training.accum)
        OmegaConf.update(cfg, "sampling.batch_per_gpu", batch_per_gpu)
    if cfg.eval.batch_size == 0:
        OmegaConf.update(cfg, "eval.batch_size", cfg.sampling.batch_per_gpu * cfg.ngpus * cfg.training.accum)

    # build sde
    token_size = train_cfg.tokens
    vocab_size = train_cfg.tokens + (1 if train_cfg.sde.prior_dist.add_mask_token else 0)
    manifold = Hypersphere(vocab_size-1)
    
    base_max_length = math.ceil(math.log(cfg.original_tokens) / math.log(token_size))
    seq_length = train_cfg.model.length * base_max_length

    mprint(f"Data {train_cfg.data.train} with base-{token_size}, base_length={base_max_length}")

    scheduler = instantiate(train_cfg.scheduler, device=device)
    prior_dist = instantiate(train_cfg.sde.prior_dist, device=device, batch_dims=(cfg.sampling.batch_per_gpu, seq_length, vocab_size))

    try:
        base_dir = os.path.dirname(os.path.dirname(cfg.model_path))
        with open(os.path.join(base_dir, "sde.pkl"), "rb") as f:
            alphas, rhos = pickle.load(f)

        sde = instantiate(
            train_cfg.sde, manifold=manifold, scheduler=scheduler, prior_dist=prior_dist,
            preprocessed=(alphas.to(device), rhos.to(device))
        )
        print("Loaded preprocessed sde.")

    except:
        print("No preprocessed sde found.")
        sde = instantiate(
            cfg.sde, manifold=manifold, scheduler=scheduler, prior_dist=prior_dist,
            device=device
        )

    # build model
    drift_model = instantiate(train_cfg.model).to(device)

    # checkpoint for torch compiled model
    if "_orig_mod" in [*loaded_state['model']][0]:
        drift_model = torch.compile(drift_model)
        drift_model.load_state_dict(loaded_state["model"])
    else:
        drift_model.load_state_dict(loaded_state["model"])
        drift_model = torch.compile(drift_model)
        
    drift_model = DDP(drift_model, device_ids=[rank]).eval()

    ema = ExponentialMovingAverage(
        drift_model.parameters(), decay=train_cfg.training.ema
    )
    ema.load_state_dict(loaded_state["ema"])    
    step = int(loaded_state['step'])

    # load ema parameters
    ema.store(drift_model.parameters())
    ema.copy_to(drift_model.parameters())

    # load in tokenizer
    tokenizer = data.get_tokenizer(cfg.data.train)

    # define sampling function
    sampling_eps = 1e-5
    sampling_shape = (cfg.sampling.batch_per_gpu, seq_length, vocab_size)
    sampling_fn = sampling.get_sampling_fn(
        cfg, sde, sampling_shape, sampling_eps, device, proj_fn=(lambda x: x)
    )

    mprint(f"Generating text at step: {step}")
    this_sample_dir = os.path.join(sample_dir, f"iter_{step}")
    utils.makedirs(this_sample_dir)

    num_samples = 4000 if cfg.data.train=='text8' else 128 #512
    samples = []
    for i in tqdm(range(math.ceil(num_samples / (sampling_shape[0] * world_size))), leave=False):
        sample = sampling_fn(drift_model).detach().cpu() # Shape: B x (mxL) x V
        samples.append(sample)

    gc.collect()
    torch.cuda.empty_cache()

    samples = torch.cat(samples, dim=0)

    shift_and_decode = sutils.find_bos_and_shift_fn(token_size, base_max_length, tokenizer)
    sentences, samples = shift_and_decode(samples)
    
    file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
    with open(file_name, 'w', encoding='utf8') as file:
        for sentence in sentences:
            file.write(sentence + "\n")
            file.write("="*100+"\n")

    with open(os.path.join(this_sample_dir, f"samples_{rank}.pkl"), 'wb') as file:
        pickle.dump(samples, file, protocol=pickle.HIGHEST_PROTOCOL)

    dist.barrier()

    if rank == 0:
        # Evaluation performed on single GPU due to measuring entropy.
        # May be modified to perform evaluation on multiple GPUs.
        evaluator = Eval(cfg, sde, distributed=False)

        samples = []
        for i in range(world_size):
            with open(os.path.join(this_sample_dir, f"samples_{i}.pkl"), 'rb') as file:
                sample = pickle.load(file)
            # samples.append(sample)
            samples.extend(sample)
        # samples = torch.cat(samples)

        print(f"{len(samples)} samples loaded.")

        result_dict = evaluator.eval(sample=samples, drift_model=drift_model)
        for k, v in result_dict.items():
            mprint(f"Step {step}. {k:8s}: {v:.3f}")

    dist.barrier()