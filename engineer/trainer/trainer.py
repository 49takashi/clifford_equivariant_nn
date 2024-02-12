import datetime
import os
import subprocess
import time
import warnings
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
import shutil
import numpy as np

from ..callbacks.checkpoint import Checkpoint
from ..loggers.loggers import ConsoleLogger

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))

import pdb

def human_format(num: float):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )

# def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
# refer https://medium.com/analytics-vidhya/saving-and-loading-your-model-to-resume-training-in-pytorch-cb687352fa61
def save_ckp(state, epoch):
    f_path = checkpoint_dir / 'checkpoint.pt'"./results/model_{:06d}.pt".format(self.current_epoch)
    torch.save(state, f_path)
    # if is_best:
    #     best_fpath = best_model_dir / 'best_model.pt'
    #     shutil.copyfile(f_path, best_fpath)

def count_parameters(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def to_device(input, device, detach=True):
    if type(input) == int:
        return input
    if isinstance(input, torch.Tensor):
        input = input.to(device)
        if detach:
            input = input.detach()
        return input
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list):
        keys = range(len(input))
    elif isinstance(input, dict):
        keys = input.keys()
    else:
        raise ValueError(f"Unknown input type {type(input)}.")
    for k in keys:
        input[k] = to_device(input[k], device)
    return input


def run_bash_command(command: str) -> str:
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )

    if result.returncode == 0:
        output = result.stdout.strip()
        return output
    else:
        error = result.stderr.strip()
        raise RuntimeError(f"Error executing command: {error}")


def parse_time_components(time_string: str):
    days, hours, minutes, seconds = 0, 0, 0, 0

    # Splitting days if present.
    if "-" in time_string:
        try:
            days_str, time_string = time_string.split("-")
        except:
            raise ValueError(f"Invalid time format {time_string}.")
        days = int(days_str)

    # Splitting hours, minutes, and seconds.
    time_components = time_string.split(":")
    num_components = len(time_components)

    if num_components == 3:
        hours, minutes, seconds = map(int, time_components)
    elif num_components == 2:
        minutes, seconds = map(int, time_components)
    elif num_components == 1:
        seconds = int(time_components[0])
    else:
        raise ValueError(f"Invalid time format {time_string}.")

    return days, hours, minutes, seconds


def parse_slurm_time(time_string) -> datetime.timedelta:
    days, hours, minutes, seconds = parse_time_components(time_string)
    return datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def _parse_max_time(time):
    if time is None:
        return

    if time is None and "SLURM_JOB_ID" in os.environ:
        time = run_bash_command(
            "squeue -j $SLURM_JOB_ID -h --Format TimeLimit"
        ).splitlines()
        if len(time) > 1:
            warnings.warn(
                "More than one job found (array job?). Using the first one for setting the time limit."
            )
        time = time[0]

    max_time = parse_slurm_time(time)
    return max_time


class Trainer:
    def __init__(
        self,
        scheduler=None,
        logger: Any = None,
        max_steps: int = 0,
        max_time: str = None,
        limit_val_batches: int = float("inf"),
        val_check_interval: int = 1024,
        print_interval: int = 32,
        fast_dev_run: bool = False,
        wandb=None,
        callbacks=list(),
        log_interval=256,
        checkpoint=None,
        test_only=False,
        data_type=None,
        action=None,
    ):
        self.callbacks = callbacks

        if logger is None:
            if wandb:
                logger = WANDBLogger()
            else:
                logger = ConsoleLogger()

        if any(isinstance(c, Checkpoint) for c in callbacks):
            assert (
                checkpoint is None
            ), f"Checkpoint {checkpoint} is already in callbacks."
            checkpoint = next(c for c in callbacks if isinstance(c, Checkpoint))
        elif checkpoint is None and not any(
            isinstance(c, Checkpoint) for c in callbacks
        ):
            checkpoint = Checkpoint("val/loss")
            callbacks.append(checkpoint)
        elif isinstance(checkpoint, str):
            checkpoint = Checkpoint(dir=checkpoint)
        else:
            raise ValueError(f"Unknown checkpoint: {checkpoint}.")

        if fast_dev_run:
            print("This is a development run. Limiting the number of batches to 1.")
            max_steps = 1
            limit_val_batches = 1

        self.starting_time = datetime.datetime.now()
        self.max_time = _parse_max_time(max_time)
        self.checkpoint = checkpoint
        self.fast_dev_run = fast_dev_run
        self.scheduler = scheduler
        self.max_steps = max_steps
        self.limit_val_batches = limit_val_batches
        self.val_check_interval = val_check_interval
        self.logger = logger
        self.print_interval = print_interval
        self.log_interval = log_interval
        self.test_only = test_only
        assert data_type in ["nbody_multi", "protein", "motioncap", "md17"]
        self.data_type = data_type
        if data_type == "motioncap":
            assert action in ["run", "walk"]
            self.action = action

        self.is_stacked=False
        self.stack_adjacent = None
        
        self.is_distributed = dist.is_initialized()

        self.global_step = 0
        self.current_epoch = 0

        self.should_raise = None
        self.should_test = False
        self.device = None

        self.test_log = []

    def _add_prefix(
        self, metrics: dict[str, torch.Tensor], prefix: str
    ) -> dict[str, torch.Tensor]:
        return {f"{prefix}/{k}": v for k, v in metrics.items()}

    def train_step(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, batch: Any
    ):
        model.train()

        batch = to_device(batch, self.device)

        # pdb.set_trace()
        if "Dense" in model.__class__.__name__ and (self.data_type == "protein" or self.data_type == "motioncap" or "nbody_multi"):
            loss, mse_loss, mse_outputs = model.loss_after_forward(batch, self.global_step)
        # elif "NBodyCGGNN" in model.__class__.__name__:
        #     loss, mse_outputs = model.forward(batch, self.global_step)
        else:    
            loss, mse_outputs = model.loss_after_forward(batch, self.global_step)
            # print(loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if torch.isnan(loss):
            self.should_raise = ValueError("Loss is NaN.")

        if self.is_distributed:
            model.module.train_metrics.update(**mse_outputs)
        else:
            model.train_metrics.update(**mse_outputs)

        if self.global_step % self.print_interval == 0:
            if "Dense" in model.__class__.__name__ and (self.data_type == "protein" or self.data_type == "motioncap" or "nbody_multi"):
                print(f"Step: {self.global_step} (Training) FrobeniusLoss: {loss:.4f} MSELoss: {mse_loss:.4f}")
            else:    
                print(f"Step: {self.global_step} (Training) MSELoss: {loss:.4f}")

    @torch.no_grad()
    def test_loop(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        test_loader: DataLoader,
        validation=False,
    ):
        model.eval()

        num_iterations = int(min(len(test_loader), self.limit_val_batches))
        print("num iterations: ", num_iterations, self.global_step)
        t0 = time.time()

        if self.is_distributed:
            assert model.module.test_metrics.empty()
        else:
            assert model.test_metrics.empty()
        if validation:
            print_str = "Validation"
            prefix = "val"
        else:
            print_str = "Testing"
            prefix = "test"

        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= self.limit_val_batches:
                break

            batch = to_device(batch, self.device)
            if "Dense" in model.__class__.__name__ and (self.data_type == "protein" or self.data_type == "motioncap" or "nbody_multi"):
                eval_loss, eval_outputs, adjs = model.eval_after_forward(batch, batch_idx)
                loss, _, _ = model.loss_after_forward(batch, batch_idx)
            else:    
                loss, outputs = model.loss_after_forward(batch, batch_idx)

            if "Dense" in model.__class__.__name__ and (self.data_type == "protein" or self.data_type == "motioncap" or "nbody_multi"):       
                if self.is_distributed:
                    model.module.test_metrics.update(**eval_outputs)
                else:
                    model.test_metrics.update(**eval_outputs)
            else:
                if self.is_distributed:
                    model.module.test_metrics.update(**outputs)
                else:
                    model.test_metrics.update(**outputs)

            if batch_idx % self.print_interval == 0:
                print(
                    f"Step: {self.global_step} ({print_str}) Batch: {batch_idx} / {num_iterations}"
                )
                if "Dense" in model.__class__.__name__ and (self.data_type == "protein" or self.data_type == "motioncap" or "nbody_multi"):
                    print("MSELoss, FrobeniousLoss: ", eval_loss, loss)
                    # print(adjs[0][0], adjs[1][0])
                    
        t1 = time.time()
        s_it = (t1 - t0) / num_iterations

        if self.is_distributed:
            metrics = model.module.test_metrics.compute()
            if not validation and self.data_type == "protein":
                self.test_log.append(metrics["loss"].detach().cpu().numpy())
                with open('test_log_protein_denseunet_loc_innerlayer2_unit14_cat_dist_cl20_lam1_decay-8.npy', 'wb') as f:
                    np.save(f, np.array(self.test_log))
                    
                if not self.is_stacked:
                    self.is_stacked = True
                    self.stack_adjacent = torch.stack(
                        (adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()),
                        dim=-1).numpy()    
                else:
                    self.stack_adjacent = np.concatenate((
                        self.stack_adjacent, 
                        torch.stack((adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1).numpy()
                    ), axis=-1)    
                with open('test_adj_protein_denseunet_loc_innerlayer2_unit14_cat_dist_cl20_lam1_decay-8.npy', 'wb') as f:
                    np.save(f, self.stack_adjacent)
            elif not validation and self.data_type == "nbody_multi":
                self.test_log.append(metrics["loss"].detach().cpu().numpy())
                M, aveN, J = self.nb_type
                if "Dense" in model.__class__.__name__:
                    results_path = model.test_result_path
                    inner_layers = model.inner_layers
                    num_clusters = model.num_clusters
                    inner_hidden_channels = model.inner_hidden_channels
                    loss_lambda = model.loss_lambda
                    with open((results_path + 'test_log_nbody_denseunet_{}_{}_{}_{}layers_cl{}_hidden{}_{}_lambda_{}.npy').format(M, aveN, J, inner_layers, num_clusters, inner_hidden_channels, self.action, loss_lambda), 'wb') as f:
                        np.save(f, np.array(self.test_log))
                    if not self.is_stacked:
                        self.is_stacked = True
                        self.stack_adjacent = torch.stack(
                            (adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1
                        ).numpy()    
                    else:
                        self.stack_adjacent = np.concatenate((
                            self.stack_adjacent, 
                            torch.stack((adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1).numpy()
                        ), axis=-1)    
                    with open((results_path + 'test_adj_nbody_denseunet_{}_{}_{}_{}layers_cl{}_hidden{}_{}.npy').format(M, aveN, J, inner_layers, num_clusters, inner_hidden_channels, self.action), 'wb') as f:
                        np.save(f, self.stack_adjacent)
                elif "UNet" in model.__class__.__name__:    
                    sum_res = model.sum_res
                    out_hidden = model.hidden_channels
                    pool = int(model.pool_ratios[0] * 10)
                    n_layers = model.n_layers
                    depth = model.depth
                    inner_hidden = model.inner_hidden_channels
                    results_path = model.test_result_path
                    silu = model.is_clact
                    is_normal = model.is_normal
                    if not os.path.exists(results_path):
                        os.makedirs(results_path)
                    with open( ( results_path +'test_log_nbody_unet_{}_{}_{}_outhid_{}_{}layers_{}units_sumres_{}_pool_{}_depth_{}_silu_{}_normal_{}.npy').format(M, aveN, J, out_hidden, n_layers, inner_hidden, sum_res, pool, depth, silu, is_normal), 'wb') as f:
                        np.save(f, np.array(self.test_log))   
                else:    
                    M, aveN, J = self.nb_type
                    out_hidden = model.hidden_features
                    n_layers = model.n_layers
                    results_path = model.test_result_path
                    if not os.path.exists(results_path):
                        os.makedirs(results_path)                    
                    with open( ( results_path +'test_log_nbody_standard_{}_{}_{}_outhid_{}_{}layers.npy').format(M, aveN, J, out_hidden, n_layers), 'wb') as f:
                        np.save(f, np.array(self.test_log))   
            elif not validation and self.data_type == "motioncap":
                self.test_log.append(metrics["loss"].detach().cpu().numpy())
                if "Dense" in model.__class__.__name__:
                    results_path = model.test_result_path
                    inner_layers = model.inner_layers
                    num_clusters = model.num_clusters
                    inner_hidden_channels = model.inner_hidden_channels
                    with open((results_path + 'test_log_motioncap_denseunet_{}layers_cl{}_hidden{}_{}.npy').format(inner_layers, num_clusters, inner_hidden_channels, self.action), 'wb') as f:
                        np.save(f, np.array(self.test_log))
                    if not self.is_stacked:
                        self.is_stacked = True
                        self.stack_adjacent = torch.stack(
                            (adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1
                        ).numpy()    
                    else:
                        self.stack_adjacent = np.concatenate((
                            self.stack_adjacent, 
                            torch.stack((adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1).numpy()
                        ), axis=-1)    
                    with open((results_path + 'test_adj_motioncap_denseunet_{}layers_cl{}_hidden{}_{}.npy').format(inner_layers, num_clusters, inner_hidden_channels, self.action), 'wb') as f:
                        np.save(f, self.stack_adjacent)
                elif "UNet" in model.__class__.__name__:    
                    sum_res = model.sum_res
                    out_hidden = model.hidden_channels
                    pool = int(model.pool_ratios[0] * 10)
                    n_layers = model.n_layers
                    depth = model.depth
                    inner_hidden = model.inner_hidden_channels
                    results_path = model.test_result_path
                    with open((results_path +'test_log_motioncap_unet_{}_outhid_{}_{}layers_{}units_sumres_{}_pool_{}_depth_{}.npy').format(self.action, out_hidden, n_layers, inner_hidden, sum_res, pool, depth), 'wb') as f:
                        np.save(f, np.array(self.test_log))
                elif "EGNN" in model.__class__.__name__:
                    in_node_nf = model.in_node_nf
                    in_edge_nf = model.in_edge_nf
                    hidden_nf = model.hidden_nf
                    n_layers = model.n_layers
                    results_path = model.test_result_path
                    with open((results_path +'test_log_motioncap_EGNN_{}_ndnf_{}_ednf_{}_hdnf_{}_layers_{}.npy').format(self.action, in_node_nf, in_edge_nf, hidden_nf, n_layers), 'wb') as f:
                        np.save(f, np.array(self.test_log))
                else:
                    out_hidden = model.hidden_features
                    n_layers = model.n_layers
                    results_path = model.test_result_path
                    with open((results_path +'test_log_motioncap_standard_{}_outhid_{}_{}layers.npy').format(self.action, out_hidden, n_layers), 'wb') as f:
                        np.save(f, np.array(self.test_log))
            elif not validation and self.data_type == "md17":
                self.test_log.append(metrics["loss"].detach().cpu().numpy())
                if "Dense" in model.__class__.__name__:
                    with open('test_log_md17_denseunet_2layers_cl5_hidden14.npy', 'wb') as f:
                        np.save(f, np.array(self.test_log))
                    if not self.is_stacked:
                        self.is_stacked = True
                        self.stack_adjacent = torch.stack(
                            (adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1
                        ).numpy()    
                    else:
                        self.stack_adjacent = np.concatenate((
                            self.stack_adjacent, 
                            torch.stack((adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1).numpy()
                        ), axis=-1)    
                    with open('test_adj_md17_denseunet_2layers_cl5_hidden14.npy', 'wb') as f:
                        np.save(f, self.stack_adjacent)
                elif "UNet" in model.__class__.__name__:    
                    with open('test_log_md17_unet_innerlayer3_unit14_ratio05.npy', 'wb') as f:
                        np.save(f, np.array(self.test_log))
                else:
                    with open('test_log_md17_standard.npy', 'wb') as f:
                        np.save(f, np.array(self.test_log))            
            model.module.test_metrics.reset()
        else:
            metrics = model.test_metrics.compute()
            if not validation and self.data_type == "protein":
                self.test_log.append(metrics["loss"].detach().cpu().numpy())
                with open('test_log_protein_denseunet_loc_innerlayer2_unit14_cat_dist_cl20_lam1_decay-8.npy', 'wb') as f:
                # with open('test_log_protein_share_layer_layer4_units28.npy', 'wb') as f:        
                    np.save(f, np.array(self.test_log))
                    
                if not self.is_stacked:
                    self.is_stacked = True
                    self.stack_adjacent = torch.stack(
                        (adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1
                    ).numpy()    
                else:
                    self.stack_adjacent = np.concatenate((
                        self.stack_adjacent, 
                        torch.stack((adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1).numpy()
                    ), axis=-1)    
                with open('test_adj_protein_denseunet_loc_innerlayer2_unit14_cat_dist_cl20_lam1_decay-8.npy', 'wb') as f:
                    np.save(f, self.stack_adjacent)
                    
            elif not validation and self.data_type == "nbody_multi":
                self.test_log.append(metrics["loss"].detach().cpu().numpy())
                M, aveN, J = self.nb_type
                if "Dense" in model.__class__.__name__:
                    results_path = model.test_result_path
                    inner_layers = model.inner_layers
                    num_clusters = model.num_clusters
                    inner_hidden_channels = model.inner_hidden_channels
                    local_hidden_channels = model.local_hidden_channels
                    local_hidden_layers = model.local_hidden_layers
                    loss_lambda = model.loss_lambda
                    hard_assignment = model.hard_assignment
                    skip_interupdate = model.skip_interupdate
                    if not os.path.exists(results_path):
                        os.makedirs(results_path)
                    with open((results_path + 'test_log_nbody_denseunet_{}_{}_{}_{}layers_cl{}_hidden{}_loclay_{}_locchan_{}_lambda_{}_hardS_{}_skipin_{}.npy'.format(M, aveN, J, inner_layers, num_clusters, inner_hidden_channels, local_hidden_layers, local_hidden_channels, loss_lambda, hard_assignment, skip_interupdate)), 'wb') as f:
                        np.save(f, np.array(self.test_log))
                    if not self.is_stacked:
                        self.is_stacked = True
                        self.stack_adjacent = torch.stack(
                            (adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1
                        ).numpy()    
                    else:
                        self.stack_adjacent = np.concatenate((
                            self.stack_adjacent, 
                            torch.stack((adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1).numpy()
                        ), axis=-1)    
                    with open((results_path + 'test_adj_nbody_denseunet_{}_{}_{}_{}layers_cl{}_hidden{}_loclay_{}_locchan_{}.npy').format(M, aveN, J, inner_layers, num_clusters, inner_hidden_channels, local_hidden_layers, local_hidden_channels), 'wb') as f:
                        np.save(f, self.stack_adjacent)
                elif "UNet" in model.__class__.__name__:    
                    M, aveN, J = self.nb_type
                    sum_res = model.sum_res
                    out_hidden = model.hidden_channels
                    pool = int(model.pool_ratios[0] * 10)
                    n_layers = model.n_layers
                    depth = model.depth
                    inner_hidden = model.inner_hidden_channels
                    results_path = model.test_result_path
                    silu = model.is_clact
                    is_normal = model.is_normal
                    use_skipconn = model.use_skipconn
                    if not os.path.exists(results_path):
                        os.makedirs(results_path)
                    with open(( results_path +'test_log_nbody_unet_{}_{}_{}_outhid_{}_{}layers_{}units_sumres_{}_pool_{}_depth_{}_silu_{}_normal_{}_skip_{}.npy').format(M, aveN, J, out_hidden, n_layers, inner_hidden, sum_res, pool, depth, silu, is_normal, use_skipconn), 'wb') as f:
                        np.save(f, np.array(self.test_log))   
                else:    
                    M, aveN, J = self.nb_type
                    out_hidden = model.hidden_features
                    n_layers = model.n_layers
                    results_path = model.test_result_path
                    if not os.path.exists(results_path):
                        os.makedirs(results_path)                    
                    with open( ( results_path +'test_log_nbody_standard_{}_{}_{}_outhid_{}_{}layers.npy').format(M, aveN, J, out_hidden, n_layers), 'wb') as f:
                        np.save(f, np.array(self.test_log))                     
            elif not validation and self.data_type == "motioncap":
                self.test_log.append(metrics["loss"].detach().cpu().numpy())                
                if "Dense" in model.__class__.__name__:
                    results_path = model.test_result_path
                    inner_layers = model.inner_layers
                    out_hidden = model.out_channels
                    num_clusters = model.num_clusters
                    inner_hidden_channels = model.inner_hidden_channels
                    local_hidden_channels = model.local_hidden_channels
                    local_hidden_layers = model.local_hidden_layers
                    hard_assignment = model.hard_assignment
                    skip_interupdate = model.skip_interupdate
                    if not os.path.exists(results_path):
                        os.makedirs(results_path)                    
                    with open((results_path + 'test_log_motioncap_denseunet_{}_{}layers_{}units_cluster_{}_lochid_{}_loclay_{}_hardS_{}_skipin_{}.npy').format(self.action, inner_layers, inner_hidden_channels, num_clusters, local_hidden_channels, local_hidden_layers, hard_assignment, skip_interupdate), 'wb') as f:
                        np.save(f, np.array(self.test_log))
                    if not self.is_stacked:
                        self.is_stacked = True
                        self.stack_adjacent = torch.stack(
                            (adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1
                        ).numpy()    
                    else:
                        self.stack_adjacent = np.concatenate((
                            self.stack_adjacent, 
                            torch.stack((adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1).numpy()
                        ), axis=-1)    
                    with open((results_path + 'test_adj_motioncap_denseunet_{}layers_cl{}_hidden{}_{}_hardS_{}_skipin_{}.npy').format(inner_layers, num_clusters, inner_hidden_channels, self.action, hard_assignment, skip_interupdate), 'wb') as f:
                        np.save(f, self.stack_adjacent)
                elif "UNet" in model.__class__.__name__:    
                    sum_res = model.sum_res
                    out_hidden = model.hidden_channels
                    pool = int(model.pool_ratios[0] * 10)
                    n_layers = model.n_layers
                    depth = model.depth
                    inner_hidden = model.inner_hidden_channels
                    results_path = model.test_result_path
                    if not os.path.exists(results_path):
                        os.makedirs(results_path)
                    with open((results_path +'test_log_motioncap_unet_{}_outhid_{}_{}layers_{}units_sumres_{}_pool_{}_depth_{}.npy').format(self.action, out_hidden, n_layers, inner_hidden, sum_res, pool, depth), 'wb') as f:
                        np.save(f, np.array(self.test_log))
                elif "EGNN" in model.__class__.__name__:
                    in_node_nf = model.in_node_nf
                    in_edge_nf = model.in_edge_nf
                    hidden_nf = model.hidden_nf
                    n_layers = model.n_layers
                    results_path = model.test_result_path
                    if not os.path.exists(results_path):
                        os.makedirs(results_path)
                    with open((results_path +'test_log_motioncap_EGNN_{}_ndnf_{}_ednf_{}_hdnf_{}_layers_{}.npy').format(self.action, in_node_nf, in_edge_nf, hidden_nf, n_layers), 'wb') as f:
                        np.save(f, np.array(self.test_log))
                else:
                    out_hidden = model.hidden_features
                    n_layers = model.n_layers
                    results_path = model.test_result_path
                    if not os.path.exists(results_path):
                        os.makedirs(results_path)
                    with open((results_path +'test_log_motioncap_standard_{}_outhid_{}_{}layers.npy').format(self.action, out_hidden, n_layers), 'wb') as f:
                        np.save(f, np.array(self.test_log))                        
            elif not validation and self.data_type == "md17":
                self.test_log.append(metrics["loss"].detach().cpu().numpy())
                if "Dense" in model.__class__.__name__:
                    with open('test_log_md17_denseunet_2layers_cl5_hidden14.npy', 'wb') as f:
                        np.save(f, np.array(self.test_log))
                    if not self.is_stacked:
                        self.is_stacked = True
                        self.stack_adjacent = torch.stack(
                            (adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1
                        ).numpy()    
                    else:
                        self.stack_adjacent = np.concatenate((
                            self.stack_adjacent, 
                            torch.stack((adjs[0][0].detach().cpu(), adjs[1][0].detach().cpu()), dim=-1).numpy()
                        ), axis=-1)    
                    with open('test_adj_md17_denseunet_2layers_cl5_hidden14.npy', 'wb') as f:
                        np.save(f, self.stack_adjacent)
                elif "UNet" in model.__class__.__name__:    
                    with open('test_log_md17_unet_innerlayer3_unit14_ratio05.npy', 'wb') as f:
                        np.save(f, np.array(self.test_log))
                else:
                    with open('test_log_md17_standard.npy', 'wb') as f:
                        np.save(f, np.array(self.test_log)) 
            model.test_metrics.reset()
        metrics[f"s_it"] = s_it

        metrics = self._add_prefix(metrics, prefix)

        if self.logger:
            self.logger.log_metrics(metrics, step=self.global_step)

        if validation:
            for callback in self.callbacks:
                callback.on_test_end(self, model, optimizer, metrics)

    @property
    def should_stop(self):
        if (
            self.max_time is not None
            and self.max_time < datetime.datetime.now() - self.starting_time
        ):
            print("Stopping due to max_time.")
            return True
        if self.max_steps is not None and self.global_step >= self.max_steps:
            print("Stopping due to max_steps.")
            return True
        return False

    def fit(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader,
        val_loader=None,
        test_loader=None,
        nb_type=None,
    ):
        self.nb_type = nb_type
        if hasattr(model, "device"):
            device = model.device
            if type(device) == str:
                device = next(model.parameters()).device
        else:
            device = next(model.parameters()).device
        self.device = device

        if torch.cuda.is_available() and not device.type == "cuda":
            warnings.warn("CUDA is available but not being used.")

        print("\nModel Summary\n---")
        print(model)
        print(f"Total parameters: {human_format(count_parameters(model))}\n")

        if self.checkpoint:
            self.checkpoint.restore(self, model, optimizer)

        if self.test_only:
            print(f"Testing mode.")
            with torch.no_grad():
                self.test_loop(model, optimizer, test_loader, validation=False)
            return

        t0 = time.time()

        last_global_step = self.global_step

        while not self.should_stop:
            if self.is_distributed:
                train_loader.sampler.set_epoch(self.current_epoch)
            for batch in train_loader:
                self.train_step(model, optimizer, batch)

                if self.scheduler is not None:
                    self.scheduler.step()

                lr = optimizer.param_groups[0]["lr"]

                if self.global_step % self.log_interval == 0:
                    t1 = time.time()
                    if self.is_distributed:
                        train_metrics = model.module.train_metrics.compute()
                        model.module.train_metrics.reset()
                    else:
                        train_metrics = model.train_metrics.compute()
                        model.train_metrics.reset()

                    s_it = (t1 - t0) / (self.global_step + 1 - last_global_step)
                    train_metrics["s_it"] = s_it
                    train_metrics["lr"] = lr
                    train_metrics["epoch"] = self.current_epoch

                    if self.logger:
                        train_metrics = self._add_prefix(train_metrics, "train")
                        self.logger.log_metrics(train_metrics, step=self.global_step)

                    t0 = time.time()
                    last_global_step = self.global_step

                if self.global_step % self.val_check_interval == 0:
                    if val_loader is not None and self.limit_val_batches > 0:
                        with torch.no_grad():
                            self.test_loop(
                                model, optimizer, val_loader, validation=True
                            )

                    t0 = time.time()
                    last_global_step = self.global_step

                    if self.should_test:
                        if test_loader is not None:
                            with torch.no_grad():
                                self.test_loop(
                                    model, optimizer, test_loader, validation=False
                                )
                                self.should_test = False

                self.global_step += 1

                if self.should_raise is not None:
                    raise self.should_raise


                if self.should_stop:
                    if self.data_type=="nbody_multi":
                        if "Dense" in model.__class__.__name__:
                            if type(nb_type)==tuple:
                                M, aveN, J = nb_type
                                results_path = model.test_result_path
                                inner_layers = model.inner_layers
                                num_clusters = model.num_clusters
                                local_hidden_layers = model.local_hidden_layers
                                local_hidden_channels = model.local_hidden_channels
                                inner_hidden_channels = model.inner_hidden_channels
                                loss_lambda = model.loss_lambda
                                hard_assignment = model.hard_assignment
                                skip_interupdate = model.skip_interupdate

                                path = './results/multi_nbody_model_final_denseunet_{}_{}_{}_{}layers_cl{}_hidden{}_loclay_{}_locchan_{}_lambda_{}_hardS_{}_skipin_{}.pth'.format(M, aveN, J, inner_layers, num_clusters, inner_hidden_channels, local_hidden_layers, local_hidden_channels, loss_lambda, hard_assignment, skip_interupdate)
                                torch.save(model.state_dict(), path)
                            else:
                                path = "./results/multi_nbody_denseunetmodel_final_temp.pth"
                                torch.save(model.state_dict(), path)
                        elif "UNet" in model.__class__.__name__:
                            if type(nb_type)==tuple:
                                M, aveN, J = nb_type
                                out_hidden = model.hidden_channels
                                sum_res = model.sum_res
                                pool = int(model.pool_ratios[0] * 10)
                                n_layers = model.n_layers
                                depth = model.depth
                                inner_hidden = model.inner_hidden_channels
                                silu = model.is_clact
                                is_normal = model.is_normal
                                use_skipconn = model.use_skipconn
                                path = "./results/multi_nbody_model_final_unet_{}_{}_{}_{}layers_outhid_{}_{}units_sumres_{}_pool_{}_depth_{}_silu_{}_normal_{}_skip_{}.pth".format(M, aveN, J, n_layers, out_hidden, inner_hidden, sum_res, pool, depth, silu, is_normal, use_skipconn)
                                torch.save(model.state_dict(), path)
                            else:
                                path = "./results/multi_nbody_unetmodel_final_temp.pth"
                                torch.save(model.state_dict(), path)
                        else:
                            if type(nb_type)==tuple:
                                M, aveN, J = nb_type
                                out_hidden = model.hidden_features
                                n_layers = model.n_layers
                                path = "./results/multi_nbody_model_final_{}_{}_{}_{}layers_{}units.pth".format(M, aveN, J, out_hidden, n_layers)
                                torch.save(model.state_dict(), path)
                            else:
                                path = "./results/multi_nbody_model_final_temp.pth"
                                torch.save(model.state_dict(), path)
                    elif self.data_type=="protein":
                        if "UNet" in model.__class__.__name__:
                            path = "./results/protein_denseunetmodel_final_loc_innerlayer2_unit14_cat_dist_cl20_lam1_decay-8.pth"
                            torch.save(model.state_dict(), path)
                        else:
                            path = "./results/protein_model_final_share_layer_layer4_units28.pth"
                            torch.save(model.state_dict(), path)
                    elif self.data_type == "motioncap":
                        if "Dense" in model.__class__.__name__:
                            inner_layers = model.inner_layers
                            num_clusters = model.num_clusters
                            inner_hidden_channels = model.inner_hidden_channels
                            local_hidden_layers = model.local_hidden_layers
                            local_hidden_channels = model.local_hidden_channels
                            path = "./results/motioncap_denseunet_final_{}_{}layers_cl{}_hidden{}_loclay_{}_locf_{}.pth".format(self.action, inner_layers, num_clusters, inner_hidden_channels, local_hidden_layers, local_hidden_channels)
                            torch.save(model.state_dict(), path)
                        elif "UNet" in model.__class__.__name__:
                            sum_res = model.sum_res
                            out_hidden = model.hidden_channels
                            pool = int(model.pool_ratios[0] * 10)
                            n_layers = model.n_layers
                            depth = model.depth
                            inner_hidden = model.inner_hidden_channels
                            path = "./results/motioncap_unet_final_{}_outhid_{}_{}layers_{}units_sumres_{}_pool_{}_depth_{}.pth".format(self.action, out_hidden, n_layers, inner_hidden, sum_res, pool, depth)
                            torch.save(model.state_dict(), path)
                        elif "EGNN" in model.__class__.__name__:
                            in_node_nf = model.in_node_nf
                            in_edge_nf = model.in_edge_nf
                            hidden_nf = model.hidden_nf
                            n_layers = model.n_layers
                            path = './results/test_log_motioncap_EGNN_{}_ndnf_{}_ednf_{}_hdnf_{}_layers_{}.pth'.format(self.action, in_node_nf, in_edge_nf, hidden_nf, n_layers)
                            torch.save(model.state_dict(), path)
                        else:
                            out_hidden = model.hidden_features
                            n_layers = model.n_layers
                            path = "./results/mocap_model_final_standard_{}_outhid_{}_{}layers.pth".format(self.action, out_hidden, n_layers)
                            torch.save(model.state_dict(), path)
                    elif self.data_type == "md17":
                        if "Dense" in model.__class__.__name__:
                            path = "./results/md17_denseunet_final_loc_innerlayer2_cl5_hidden14.pth"
                            torch.save(model.state_dict(), path)
                        elif "UNet" in model.__class__.__name__:
                            path = "./results/md17_unet_final_loc_innerlayer3_unit14_ratio05.pth"
                            torch.save(model.state_dict(), path)
                        else:
                            path = "./results/md17_model_final_standard.pth"
                            torch.save(model.state_dict(), path)
                            
                    break

            self.current_epoch += 1


