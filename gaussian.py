import hemcee
from hemcee.tests.distribution import make_gaussian, _make_covariance_skewed

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", False)

from configuration import parse_args, make_sampler, plot_samples

# RNG Settings will be set from command line arguments

# Distribution Settings
dim = 12
condition_number = 100

# Sampler Settings
total_chains = dim * 2
warmup = int(2 * 10**5)
num_samples = int(10**6)
thin_by = 10


if __name__ == "__main__":
    import sys
    import time
    
    args = parse_args()
    
    print("="*60)
    print("GAUSSIAN DISTRIBUTION MCMC SAMPLING")
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
    print(f"Total chains: {total_chains}")
    print(f"Warmup samples: {warmup}")
    print(f"Main samples: {num_samples}")
    print(f"Thin by: {thin_by}")
    print(f"Seed: {args.seed}")
    print("="*60)
    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, 3)

    print("Generating true distribution parameters...")
    true_mean = jax.random.normal(keys[1], shape=(dim,))
    true_precision = _make_covariance_skewed(keys[0], dim, condition_number)
    
    log_prob = make_gaussian(true_precision, true_mean)
    print("✓ Distribution parameters generated")

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

    print("Initializing chains...")
    initial_state = jax.random.normal(keys[1], shape=(total_chains, dim))
    print("✓ Chains initialized")
    
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
    
    print('Truth vs Empirical:')
    empirical_mean = jnp.mean(samples, axis=(0,1))
    empirical_cov = jnp.cov(samples.reshape(-1, dim).T)
    
    mean_residual = jnp.linalg.norm(true_mean - empirical_mean)
    cov_residual = jnp.linalg.norm(jnp.linalg.inv(true_precision) - empirical_cov)

    print(f"  Mean residual: {mean_residual}")
    print(f"  Covariance residual: {cov_residual}")
    
    # Generate corner plot if requested
    if args.plot:
        title = f"Gaussian Distribution - {args.move}"
        filename = f"gaussian_samples_corner_{args.move}_s{args.hamiltonian_step_size}_L{args.hamiltonian_L}"
        plot_samples(samples, dim, title, filename)
    
    
    
    