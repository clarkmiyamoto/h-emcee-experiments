import argparse

import hemcee
from hemcee.moves.hamiltonian.hmc_walk import hmc_walk_move
from hemcee.moves.hamiltonian.hmc_side import hmc_side_move
from hemcee.moves.vanilla.stretch import stretch_move
from hemcee.moves.vanilla.walk import walk_move
from hemcee.moves.vanilla.side import side_move

import jax
import jax.numpy as jnp

cpu_device = jax.devices('cpu')[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--move', type=str,
                        choices=['hmc', 'hmc_walk', 'hmc_side', 'stretch', 'walk', 'side'],
                        help='Type of move to use.')
    parser.add_argument('--hamiltonian_step_size', type=float, default=0.1,
                        help='Step size of the leapfrog integrator.')
    parser.add_argument('--hamiltonian_L', type=int, default=10,
                        help='Number of leapfrog steps per sample.')
    args = parser.parse_args()
    return args

def make_sampler(move_type,
                 total_chains,
                 dim,
                 log_prob,
                 step_size,
                 L):
    if move_type == 'hmc':
        return hemcee.HamiltonianSampler(total_chains=total_chains,
                                         dim=dim,
                                         log_prob=log_prob,
                                         step_size=step_size,
                                         inv_mass_matrix=jnp.eye(dim),
                                         L=L,
                                         storage_device=cpu_device)
    elif move_type == 'hmc_walk':
        return hemcee.HamiltonianEnsembleSampler(total_chains=total_chains,
                                                 dim=dim,
                                                 log_prob=log_prob,
                                                 step_size=step_size,
                                                 L=L,
                                                 move=hmc_walk_move,
                                                 storage_device=cpu_device)
    elif move_type == 'hmc_side':
        return hemcee.HamiltonianEnsembleSampler(total_chains=total_chains,
                                                 dim=dim,
                                                 log_prob=log_prob,
                                                 step_size=step_size,
                                                 L=L,
                                                 move=hmc_side_move,
                                                 storage_device=cpu_device)
    elif move_type == 'stretch':
        return hemcee.EnsembleSampler(total_chains=total_chains,
                                      dim=dim,
                                      log_prob=log_prob,
                                      move=stretch_move,
                                      storage_device=cpu_device)
    elif move_type == 'walk':
        return hemcee.EnsembleSampler(total_chains=total_chains,
                                      dim=dim,
                                      log_prob=log_prob,
                                      move=walk_move,
                                      storage_device=cpu_device)
    elif move_type == 'side':
        return hemcee.EnsembleSampler(total_chains=total_chains,
                                      dim=dim,
                                      log_prob=log_prob,
                                      move=side_move,
                                      storage_device=cpu_device)
    else:
        raise ValueError(f"Unknown move type: {move_type}")
    
def monitor_gpu():
    # GPU Tracking
    import wandb
    wandb.init(project="gpu-monitor", config={"purpose": "hardware monitoring"})