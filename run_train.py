import datetime
import os
import os.path
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import pickle
from functools import partial
import math

import data
import losses
import sampling
import utils.utils as utils
import utils.seq_utils as sutils
from model.ema import ExponentialMovingAverage

from hypersphere import Hypersphere
from hydra.utils import instantiate

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)

import wandb
from evaluation import Eval


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

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
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "run.log"))
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)

    if rank == 0 and cfg.use_wandb:
        wandb.init(project=f'discrete-{cfg.data.train}', name=cfg.wandb_run_name)
        wandb.config = cfg

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

    # build sde
    token_size = cfg.tokens
    vocab_size = cfg.tokens + (1 if cfg.sde.prior_dist.add_mask_token else 0)
    manifold = Hypersphere(vocab_size-1)
    
    base_max_length = math.ceil(math.log(cfg.original_tokens) / math.log(token_size))
    seq_length = cfg.model.length * base_max_length
    _base_n = partial(sutils.base_n, base=token_size, max_length=base_max_length)

    mprint(f"Data {cfg.data.train} with base-{token_size}, base_length={base_max_length}")

    scheduler = instantiate(cfg.scheduler, device=device)
    prior_dist = instantiate(cfg.sde.prior_dist, device=device, batch_dims=(cfg.training.batch_size * 2 // (cfg.ngpus * cfg.training.accum), seq_length, vocab_size))
    
    try:
        with open(os.path.join(cfg.work_dir, "sde.pkl"), "rb") as f:
            alphas, rhos = pickle.load(f)
        
        sde = instantiate(
            cfg.sde, manifold=manifold, scheduler=scheduler, prior_dist=prior_dist,
            preprocessed=(alphas.to(device), rhos.to(device))
        )
        mprint(f"Loaded preprocessed sde.")
    
    except:
        mprint(f"No preprocessed sde found.")
        sde = instantiate(
            cfg.sde, manifold=manifold, scheduler=scheduler, prior_dist=prior_dist, device=device,
        )

        if rank==0:
            with open(os.path.join(work_dir, "sde.pkl"), "wb") as f:
                pickle.dump((sde.alphas.detach().cpu(), sde.rhos.detach().cpu()), f, protocol=pickle.HIGHEST_PROTOCOL)

    if rank == 0 and len(sde.alphas.shape)==1:
        import matplotlib.pyplot as plt
        ts = np.linspace(0, 1, sde.preprocess_steps)
        fig, axs = plt.subplots(1, 2, figsize=(18,6))
        axs[0].plot(ts, sde.alphas.detach().cpu().numpy(), color='b')
        axs[0].title.set_text("alphas")
        axs[1].plot(ts, sde.rhos.detach().cpu().numpy(), color='b')
        axs[1].title.set_text("rhos")
        fig.suptitle(f"beta_0={sde.scheduler.beta_0:.1e}, beta_f={sde.scheduler.beta_f:.1e}")
        fig.savefig(os.path.join(cfg.work_dir, 'moments.png'), dpi=300, bbox_inches="tight")

    # build model
    drift_model = instantiate(cfg.model, vocab_size=vocab_size, sde=sde).to(device)
    drift_model = torch.compile(drift_model)
    drift_model = DDP(drift_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in drift_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        drift_model.parameters(), decay=cfg.training.ema
    )

    # build optimization state
    optimizer = losses.get_optimizer(cfg, drift_model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    state = dict(optimizer=optimizer, scaler=scaler, model=drift_model, sde=sde, ema=ema, step=0) 

    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg)
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # load in tokenizer
    tokenizer = data.get_tokenizer(cfg.data.train)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)

    loss_type = cfg.training.loss_type
    eval_loss_type = "elbo"
    mprint(f"Training loss type: {loss_type} | Eval loss type: {eval_loss_type} | weight type: {scheduler.weight_type}")

    train_step_fn = losses.get_step_fn(
        loss_type=loss_type, sde=sde, train=True, optimize_fn=optimize_fn, accum=cfg.training.accum
    )
    eval_step_fn = losses.get_step_fn(
        loss_type=eval_loss_type, sde=sde, train=False, optimize_fn=optimize_fn, accum=cfg.training.accum
    )

    sampling_eps = 1e-5
    if cfg.training.snapshot_sampling:
        sampling_batch_size = cfg.training.batch_size * 2 // (cfg.ngpus * cfg.training.accum)
        sampling_shape = (sampling_batch_size, seq_length, vocab_size)
        sampling_fn = sampling.get_sampling_fn(cfg, sde, sampling_shape, sampling_eps, device)
        evaluator = Eval(cfg, sde, distributed=True)

        shift_and_decode = sutils.find_bos_and_shift_fn(token_size, base_max_length, tokenizer)

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")

    while state['step'] < num_train_steps + 1:
        step = state['step']

        # Convert indices into onehot vectors
        batch = next(train_iter)['input_ids'].to(device)
        batch = _base_n(batch)
        batch = F.one_hot(batch, num_classes=vocab_size).to(torch.float32)
        
        loss = train_step_fn(state, batch)

        if step != state['step']:
            if step % cfg.training.log_freq == 0 and step > 0:
                dist.all_reduce(loss)
                loss /= world_size

                mprint("step: %d, train_loss: %.3e" % (step, loss.item()))
                if rank == 0 and cfg.use_wandb:
                    wandb.log({'train_loss' : loss.item()}, step=step)
            
            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0 and step > 0:

                # Convert indices into onehot vectors
                eval_batch = next(eval_iter)['input_ids'].to(device)
                eval_batch = _base_n(eval_batch)
                eval_batch = F.one_hot(eval_batch, num_classes=vocab_size).to(torch.float32)

                eval_loss = eval_step_fn(state, eval_batch)

                dist.all_reduce(eval_loss)
                eval_loss /= world_size

                mprint("step: %d, val_loss: %.3e" % (step, eval_loss.item()))
                if rank == 0 and cfg.use_wandb:
                    wandb.log({'val_loss' : eval_loss.item()}, step=step)

                del eval_batch

            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state
                    )

                # Generate and save samples
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    ema.store(drift_model.parameters())
                    ema.copy_to(drift_model.parameters())
                    
                    sample = sampling_fn(drift_model) # indices in base-n
                    sentences, sample = shift_and_decode(sample.detach().cpu())
                    
                    file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
                    with open(file_name, 'w', encoding='utf8') as file:
                        for sentence in sentences:
                            file.write(sentence + "\n")
                            file.write("="*100+"\n")

                    result_dict = evaluator.eval(
                        sample=sample, drift_model=drift_model
                    )
                    for k, v in result_dict.items():
                        mprint(f"Step {step}. {k:8s}: {v:.3f}")
                        if rank == 0 and cfg.use_wandb:
                            wandb.log({k : v}, step=step)

                    ema.restore(drift_model.parameters())

                dist.barrier()