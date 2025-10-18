import hemcee
import jax
import jax.numpy as jnp

from configuration import parse_args, make_sampler

# RNG Settings
seed = 0

# Distribution Settings
dim = 50
l = 0.25  # Ring thickness parameter

# Sampler Settings
total_chains = dim * 2
warmup = int(2 * 10**5)
num_samples = int(10**6)
thin_by = 10


def make_ring_distribution(l):
    """
    Create a ring distribution with probability density:
    π(x) ∝ exp(-((||x||₂² - 1)²)/l²)
    
    Parameters:
    l: float, ring thickness parameter (smaller l = thinner ring)
    
    Returns:
    log_prob: function that takes x and returns log probability
    """
    def log_prob(x):
        # Compute squared L2 norm
        norm_squared = jnp.sum(x**2)
        
        # Compute the ring distribution log probability
        # π(x) ∝ exp(-((||x||₂² - 1)²)/l²)
        # log π(x) = -((||x||₂² - 1)²)/l² + constant
        log_prob_val = -((norm_squared - 1.0)**2) / (l**2)
        
        return log_prob_val
    
    return log_prob


if __name__ == "__main__":
    args = parse_args()
    seed = 0
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)

    # Create ring distribution
    log_prob = make_ring_distribution(l)

    sampler = make_sampler(move_type=args.move,
                           total_chains=total_chains,
                           dim=dim,
                           log_prob=log_prob,
                           step_size=args.hamiltonian_step_size,
                           L=args.hamiltonian_L)

    # Initialize chains near the ring (unit sphere)
    # Start with random points on unit sphere plus small noise
    initial_directions = jax.random.normal(keys[1], shape=(total_chains, dim))
    initial_directions = initial_directions / jnp.linalg.norm(initial_directions, axis=1, keepdims=True)
    initial_noise = 0.1 * jax.random.normal(keys[2], shape=(total_chains, dim))
    initial_state = initial_directions + initial_noise
    
    samples = sampler.run_mcmc(
        key=keys[1],
        initial_state=initial_state,
        num_samples=num_samples,
        warmup=warmup,
        thin_by=thin_by,)
    

    print('Diagnostics:')
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
    
    print('Ring Distribution Analysis:')
    # Flatten samples for analysis
    flat_samples = samples.reshape(-1, dim)
    
    # Compute distances from origin
    distances = jnp.linalg.norm(flat_samples, axis=1)
    mean_distance = jnp.mean(distances)
    std_distance = jnp.std(distances)
    
    print(f"  Mean distance from origin: {mean_distance:.3f}")
    print(f"  Std distance from origin: {std_distance:.3f}")
    print(f"  Expected distance (ring radius): 1.0")
    
    # Check how many samples are within expected ring thickness
    ring_thickness = 2 * l  # Approximate thickness of the ring
    within_ring = jnp.abs(distances - 1.0) < ring_thickness
    ring_fraction = jnp.mean(within_ring)
    print(f"  Fraction within ring thickness (±{ring_thickness:.3f}): {ring_fraction:.3f}")
    
    # Compute empirical mean and covariance
    empirical_mean = jnp.mean(flat_samples, axis=0)
    empirical_cov = jnp.cov(flat_samples.T)
    
    print(f"  Empirical mean: {empirical_mean}")
    print(f"  Empirical mean magnitude: {jnp.linalg.norm(empirical_mean):.3f}")
    print(f"  Empirical covariance trace: {jnp.trace(empirical_cov):.3f}")


