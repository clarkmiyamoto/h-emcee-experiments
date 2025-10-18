import hemcee
import jax
import jax.numpy as jnp
import numpy as np

from configuration import parse_args, make_sampler

import corner

# RNG Settings
seed = 0

# Distribution Settings
dim = 2  # 2D Rosenbrock function
# Rosenbrock parameters
a = 1.0  # Standard parameter
b = 100.0  # Controls the "banana" shape curvature

# Sampler Settings
total_chains = dim * 10
warmup = 10000
num_samples = 1000
thin_by = 10


def rosenbrock_log_prob(x):
    """
    Log probability for the Rosenbrock function in 2D.
    
    The Rosenbrock function is:
    f(x,y) = (a - x)² + b(y - x²)²
    
    For MCMC, we use the negative log probability:
    -log p(x,y) ∝ f(x,y)
    
    The global minimum is at (a, a²) = (1, 1).
    
    Args:
        x: Array of shape (2,) representing [x, y] coordinates
    
    Returns:
        Log probability (negative of Rosenbrock function)
    """
    x_coord = x[0]
    y_coord = x[1]
    
    # Rosenbrock function
    rosenbrock_value = (a - x_coord)**2 + b * (y_coord - x_coord**2)**2
    
    # Return negative log probability
    return -rosenbrock_value


def create_initial_state(key, total_chains, dim):
    """
    Create initial state for the Rosenbrock function.
    Start with points distributed around the global minimum (1, 1).
    """
    # Global minimum is at (1, 1)
    # Start with points distributed around this minimum
    initial_state = jnp.ones((total_chains, dim)) + 0.5 * jax.random.normal(key, shape=(total_chains, dim))
    
    return initial_state


if __name__ == "__main__":
    args = parse_args()
    seed = 0
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)

    # Create the log probability function
    log_prob = rosenbrock_log_prob

    # Create sampler
    sampler = make_sampler(move_type=args.move,
                           total_chains=total_chains,
                           dim=dim,
                           log_prob=log_prob,
                           step_size=args.hamiltonian_step_size,
                           L=args.hamiltonian_L)

    # Create initial state
    initial_state = create_initial_state(keys[1], total_chains, dim)
    
    # Run MCMC
    samples = sampler.run_mcmc(
        key=keys[1],
        initial_state=initial_state,
        num_samples=num_samples,
        warmup=warmup,
        thin_by=thin_by,
    )
    
    print('Rosenbrock Function MCMC Diagnostics:')
    print(f'  Dimension: {dim}D')
    print(f'  Rosenbrock parameters: a={a}, b={b}')
    print(f'  Global minimum: ({a}, {a**2}) = (1, 1)')
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

    # Show plot
    _ = corner.corner(np.array(samples_flat))
    _.show()
    _.savefig("rosenbrock_samples_corner.png")
    
    # Compute statistics
    mean_sample = jnp.mean(samples_flat, axis=0)
    std_sample = jnp.std(samples_flat, axis=0)
    
    # Compute Rosenbrock function values for all samples
    rosenbrock_values = jnp.array([-log_prob(sample) for sample in samples_flat])
    
    # Distance from global minimum
    global_minimum = jnp.array([a, a**2])
    distances_from_min = jnp.linalg.norm(samples_flat - global_minimum, axis=1)
    
    print(f'  Mean sample position: [{mean_sample[0]:.3f}, {mean_sample[1]:.3f}]')
    print(f'  Std sample position: [{std_sample[0]:.3f}, {std_sample[1]:.3f}]')
    print(f'  Mean distance from global minimum: {jnp.mean(distances_from_min):.3f}')
    print(f'  Min distance from global minimum: {jnp.min(distances_from_min):.3f}')
    
    print('Rosenbrock Function Values:')
    print(f'  Mean Rosenbrock value: {jnp.mean(rosenbrock_values):.3f}')
    print(f'  Std Rosenbrock value: {jnp.std(rosenbrock_values):.3f}')
    print(f'  Min Rosenbrock value: {jnp.min(rosenbrock_values):.3f}')
    print(f'  Max Rosenbrock value: {jnp.max(rosenbrock_values):.3f}')
    
    # Check how many samples are close to the global minimum
    close_to_minimum = distances_from_min < 0.5
    print(f'  Fraction within 0.5 of global minimum: {jnp.mean(close_to_minimum):.3f}')
    
    # Sample range analysis
    x_samples = samples_flat[:, 0]
    y_samples = samples_flat[:, 1]
    
    print(f'  X coordinate range: [{jnp.min(x_samples):.3f}, {jnp.max(x_samples):.3f}]')
    print(f'  Y coordinate range: [{jnp.min(y_samples):.3f}, {jnp.max(y_samples):.3f}]')
