import argparse

import hemcee
from hemcee.moves.hamiltonian.hmc_walk import hmc_walk_move
from hemcee.moves.hamiltonian.hmc_side import hmc_side_move
from hemcee.moves.vanilla.stretch import stretch_move
from hemcee.moves.vanilla.walk import walk_move
from hemcee.moves.vanilla.side import side_move

import jax
import jax.numpy as jnp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--move', type=str,
                        choices=['hmc', 'hmc_walk', 'hmc_side', 'stretch', 'walk', 'side'],
                        help='Type of move to use.')
    parser.add_argument('--hamiltonian_step_size', type=float, default=0.1,
                        help='Step size of the leapfrog integrator.')
    parser.add_argument('--hamiltonian_L', type=int, default=10,
                        help='Number of leapfrog steps per sample.')
    parser.add_argument('--adapt_step_size', action='store_true', default=False,
                        help='Enable step size adaptation.')
    parser.add_argument('--adapt_length', action='store_true', default=False,
                        help='Enable integration length adaptation.')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Generate corner plot of samples.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility.')
    args = parser.parse_args()
    return args

def make_sampler(move_type,
                 total_chains,
                 dim,
                 log_prob,
                 step_size,
                 L,
                 adapt_step_size,
                 adapt_length):
    if move_type == 'hmc':
        return hemcee.HamiltonianSampler(total_chains=total_chains,
                                         dim=dim,
                                         log_prob=log_prob,
                                         step_size=step_size,
                                         inv_mass_matrix=jnp.eye(dim),
                                         L=L,)
    elif move_type == 'hmc_walk':
        return hemcee.HamiltonianEnsembleSampler(total_chains=total_chains,
                                                 dim=dim,
                                                 log_prob=log_prob,
                                                 step_size=step_size,
                                                 L=L,
                                                 move=hmc_walk_move,
                                                 adapt_step_size=adapt_step_size,
                                                 adapt_length=adapt_length)
    elif move_type == 'hmc_side':
        return hemcee.HamiltonianEnsembleSampler(total_chains=total_chains,
                                                 dim=dim,
                                                 log_prob=log_prob,
                                                 step_size=step_size,
                                                 L=L,
                                                 move=hmc_side_move,
                                                 adapt_step_size=adapt_step_size,
                                                 adapt_length=adapt_length)
    elif move_type == 'stretch':
        return hemcee.EnsembleSampler(total_chains=total_chains,
                                      dim=dim,
                                      log_prob=log_prob,
                                      move=stretch_move,)
    elif move_type == 'walk':
        return hemcee.EnsembleSampler(total_chains=total_chains,
                                      dim=dim,
                                      log_prob=log_prob,
                                      move=walk_move,)
    elif move_type == 'side':
        return hemcee.EnsembleSampler(total_chains=total_chains,
                                      dim=dim,
                                      log_prob=log_prob,
                                      move=side_move,)
    else:
        raise ValueError(f"Unknown move type: {move_type}")
    
def plot_samples(samples, dim, title, filename, labels=None):
    """
    Generate and save a corner plot of MCMC samples.
    
    Parameters:
    samples: JAX array of shape (num_samples, total_chains, dim)
    dim: int, dimension of the state space
    title: str, title for the plot
    filename: str, filename to save the plot (without extension)
    labels: list of str, optional labels for each dimension
    """
    print("Generating corner plot...")
    try:
        import corner
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Convert JAX arrays to numpy for corner plot
        samples_np = np.array(samples)
        
        # Use a subset of samples for plotting (last 10% to avoid memory issues)
        nshow = int(0.1 * samples_np.shape[0])
        samples_plot = samples_np[-nshow:].reshape(-1, dim)
        
        # Create default labels if not provided
        if labels is None:
            labels = [f'x{i}' for i in range(dim)]
        
        # Create corner plot
        fig = corner.corner(samples_plot, 
                          labels=labels,
                          title=title,
                          show_titles=True)
        
        # Save the plot
        plot_filename = f'{filename}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Corner plot saved as: {plot_filename}")
        
        # Show the plot
        plt.show()
        
    except ImportError:
        print("Warning: corner and matplotlib not available. Skipping plot generation.")
    except Exception as e:
        print(f"Warning: Error generating corner plot: {e}")

def monitor_gpu():
    # GPU Tracking
    import wandb
    wandb.init(project="gpu-monitor", config={"purpose": "hardware monitoring"})