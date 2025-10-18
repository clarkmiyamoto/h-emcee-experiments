import hemcee
import jax
import jax.numpy as jnp
from jax import grad, vmap
jax.config.update("jax_enable_x64", True)

from configuration import parse_args, make_sampler

# RNG Settings
seed = 0

# Distribution Settings
# Discretize the unit interval [0,1] into n_points
n_points = 128  # Number of discretization points
dim = n_points  # Dimension of the state space
dx = 1.0 / (n_points - 1)  # Grid spacing

# Sampler Settings
total_chains = dim * 2
warmup = int(2 * 10**5)
num_samples = int(10**6)
thin_by = 10


def double_well_potential(u):
    """
    Double-well potential function: V(u) = (1 - u²)²
    """
    return (1 - u**2)**2


def allen_cahn_log_prob(u):
    """
    Log probability for the Allen-Cahn equation:
    exp(-∫₀¹ (½(∂ₓu(x))² + V(u(x))) dx)
    
    where V(u) = (1 - u²)² is the double-well potential.
    
    Args:
        u: Array of shape (n_points,) representing the discretized function u(x)
    
    Returns:
        Log probability (negative energy)
    """
    # Create grid points
    x = jnp.linspace(0, 1, n_points)
    
    # Compute spatial derivative using finite differences
    # For interior points: (u[i+1] - u[i-1]) / (2*dx)
    # For boundary points, use one-sided differences
    du_dx = jnp.zeros_like(u)
    
    # Interior points (central difference)
    du_dx = du_dx.at[1:-1].set((u[2:] - u[:-2]) / (2 * dx))
    
    # Boundary points (one-sided differences)
    du_dx = du_dx.at[0].set((u[1] - u[0]) / dx)
    du_dx = du_dx.at[-1].set((u[-1] - u[-2]) / dx)
    
    # Compute the integrand: ½(∂ₓu(x))² + V(u(x))
    kinetic_term = 0.5 * du_dx**2
    potential_term = double_well_potential(u)
    integrand = kinetic_term + potential_term
    
    # Integrate using trapezoidal rule
    # The factor dx/2 accounts for the trapezoidal rule weights
    integral = dx * (0.5 * integrand[0] + jnp.sum(integrand[1:-1]) + 0.5 * integrand[-1])
    
    # Return negative log probability (energy)
    return -integral


def create_initial_state(key, total_chains, dim):
    """
    Create initial state for the Allen-Cahn equation.
    Start with small random perturbations around the stable states u = ±1.
    """
    # Create initial states near the stable points u = ±1
    # Use a mixture of states near +1 and -1
    key1, key2 = jax.random.split(key)
    
    # Half the chains start near +1, half near -1
    n_chains_per_state = total_chains // 2
    
    # States near +1
    states_plus = 1.0 + 0.1 * jax.random.normal(key1, shape=(n_chains_per_state, dim))
    
    # States near -1  
    states_minus = -1.0 + 0.1 * jax.random.normal(key2, shape=(total_chains - n_chains_per_state, dim))
    
    # Combine the states
    initial_state = jnp.vstack([states_plus, states_minus])
    
    return initial_state


if __name__ == "__main__":
    import time
    
    args = parse_args()
    
    print("="*60)
    print("ALLEN-CAHN EQUATION MCMC SAMPLING")
    print("="*60)
    print(f"Move type: {args.move}")
    print(f"Step size: {args.hamiltonian_step_size}")
    print(f"Integration length: {args.hamiltonian_L}")
    print(f"Discretization points: {n_points}")
    print(f"Grid spacing: {dx:.4f}")
    print(f"Total chains: {total_chains}")
    print(f"Warmup samples: {warmup}")
    print(f"Main samples: {num_samples}")
    print(f"Thin by: {thin_by}")
    print("="*60)
    
    seed = 0
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)

    print("Setting up Allen-Cahn equation...")
    # Create the log probability function
    log_prob = allen_cahn_log_prob
    print("✓ Log probability function created")

    print("Creating sampler...")
    # Create sampler
    sampler = make_sampler(move_type=args.move,
                           total_chains=total_chains,
                           dim=dim,
                           log_prob=log_prob,
                           step_size=args.hamiltonian_step_size,
                           L=args.hamiltonian_L)
    print("✓ Sampler created")

    print("Creating initial state...")
    # Create initial state
    initial_state = create_initial_state(keys[1], total_chains, dim)
    print("✓ Initial state created (mixture near ±1)")
    
    print("Starting MCMC sampling...")
    print(f"  Warmup phase: {warmup} samples")
    print(f"  Main phase: {num_samples} samples")
    start_time = time.time()
    
    # Run MCMC
    samples = sampler.run_mcmc(
        key=keys[1],
        initial_state=initial_state,
        num_samples=num_samples,
        warmup=warmup,
        thin_by=thin_by,
    )
    
    end_time = time.time()
    print(f"✓ MCMC sampling completed in {end_time - start_time:.2f} seconds")
    
    print('Allen-Cahn Equation MCMC Diagnostics:')
    print(f'  Discretization points: {n_points}')
    print(f'  Grid spacing: {dx:.4f}')
    print(f'  Total chains: {total_chains}')
    print(f'  Samples per chain: {num_samples}')
    
    try:
        acceptance_rate_warmup = sampler.diagnostics_warmup['acceptance_rate']
        print(f'  Average warmup acceptance rate: {jnp.mean(acceptance_rate_warmup):.3f}')
    except:
        pass
    acceptance_rate_main = sampler.diagnostics_main['acceptance_rate']
    print(f'  Average main acceptance rate: {jnp.mean(acceptance_rate_main):.3f}')

    print('Autocorrelation:')
    try:
        tau = hemcee.autocorr.integrated_time(samples)
        print(f'  Integrated autocorrelation: {tau}')
    except:
        pass
    
    print('Sample Statistics:')
    # Reshape samples for analysis
    samples_flat = samples.reshape(-1, dim)
    
    # Compute statistics
    mean_sample = jnp.mean(samples_flat, axis=0)
    std_sample = jnp.std(samples_flat, axis=0)
    
    # Check if samples are near the stable states ±1
    samples_near_plus1 = jnp.mean(jnp.abs(samples_flat - 1.0), axis=1) < 0.5
    samples_near_minus1 = jnp.mean(jnp.abs(samples_flat + 1.0), axis=1) < 0.5
    
    print(f'  Mean value across all samples: {jnp.mean(mean_sample):.3f}')
    print(f'  Std value across all samples: {jnp.mean(std_sample):.3f}')
    print(f'  Fraction near +1: {jnp.mean(samples_near_plus1):.3f}')
    print(f'  Fraction near -1: {jnp.mean(samples_near_minus1):.3f}')
    
    # Compute energy statistics
    energies = -jnp.array(log_prob(samples_flat))
    print(f'  Mean energy: {jnp.mean(energies):.3f}')
    print(f'  Std energy: {jnp.std(energies):.3f}')
    print(f'  Min energy: {jnp.min(energies):.3f}')
    print(f'  Max energy: {jnp.max(energies):.3f}')
