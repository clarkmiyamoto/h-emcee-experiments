import hemcee
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", False)

from configuration import parse_args, make_sampler, plot_samples

# RNG Settings will be set from command line arguments

# Distribution Settings
dim = 12
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
    import time
    
    args = parse_args()
    
    print("="*60)
    print("RING DISTRIBUTION MCMC SAMPLING")
    print("="*60)
    print(f"Move type: {args.move}")
    
    # Autotuning information
    if args.move in ['hmc', 'hmc_walk', 'hmc_side']:
        print(f"Step size adaptation: {'Enabled' if args.adapt_step_size else 'Disabled'}")
        print(f"Integration length adaptation: {'Enabled' if args.adapt_length else 'Disabled'}")
    else:
        print("Autotuning: Not available for this move type")
    
    print(f"Step size: {args.hamiltonian_step_size}")
    print(f"Integration length: {args.hamiltonian_L}")
    print(f"Dimension: {dim}")
    print(f"Ring thickness parameter: {l}")
    print(f"Total chains: {total_chains}")
    print(f"Warmup samples: {warmup}")
    print(f"Main samples: {num_samples}")
    print(f"Thin by: {thin_by}")
    print(f"Seed: {args.seed}")
    print("="*60)
    
    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, 3)

    print("Creating ring distribution...")
    # Create ring distribution
    log_prob = make_ring_distribution(l)
    print("✓ Ring distribution created")

    print("Creating sampler...")
    sampler = make_sampler(move_type=args.move,
                           total_chains=total_chains,
                           dim=dim,
                           log_prob=log_prob,
                           step_size=args.hamiltonian_step_size,
                           L=args.hamiltonian_L,
                           adapt_step_size=args.adapt_step_size,
                           adapt_length=args.adapt_length)
    print("✓ Sampler created")

    print("Initializing chains near unit sphere...")
    # Initialize chains near the ring (unit sphere)
    # Start with random points on unit sphere plus small noise
    initial_directions = jax.random.normal(keys[1], shape=(total_chains, dim))
    initial_directions = initial_directions / jnp.linalg.norm(initial_directions, axis=1, keepdims=True)
    initial_noise = 0.1 * jax.random.normal(keys[2], shape=(total_chains, dim))
    initial_state = initial_directions + initial_noise
    print("✓ Chains initialized near unit sphere")
    
    print("Starting MCMC sampling...")
    print(f"  Warmup phase: {warmup} samples")
    print(f"  Main phase: {num_samples} samples")
    start_time = time.time()
    
    samples = sampler.run_mcmc(
        key=keys[1],
        initial_state=initial_state,
        num_samples=num_samples,
        warmup=warmup,
        thin_by=thin_by,
        show_progress=True)
    
    end_time = time.time()
    print(f"✓ MCMC sampling completed in {end_time - start_time:.2f} seconds")
    

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
        
        # Calculate autocorrelation time per total compute time
        total_compute_time = end_time - start_time
        tau_over_time = tau / total_compute_time
        print(f'  Autocorrelation time / total compute time: {tau_over_time:.6f}')
        
        # Calculate Effective Sample Size (EES)
        # EES = num_samples * total_chains / tau
        total_samples = num_samples * total_chains
        ees = total_samples / tau
        print(f'  EES (Effective Sample Size): {ees:.2f}')
        
        # Calculate EES per total compute time
        ees_per_time = ees / total_compute_time
        print(f'  EES / total compute time: {ees_per_time:.6f}')
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
    
    # Generate corner plot if requested
    if args.plot:
        title = f"Ring Distribution - {args.move}"
        filename = f"ring_samples_corner_{args.move}_s{args.hamiltonian_step_size}_L{args.hamiltonian_L}"
        plot_samples(samples, dim, title, filename)


