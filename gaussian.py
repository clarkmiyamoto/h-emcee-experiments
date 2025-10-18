import hemcee
from hemcee.tests.distribution import make_gaussian, _make_covariance_skewed

import jax
import jax.numpy as jnp

from configuration import parse_args, make_sampler

# RNG Settings
seed = 0

# Distribution Settings
dim = 128
condition_number = 1000

# Sampler Settings
total_chains = dim * 2
warmup = int(2 * 10**5)
num_samples = int(10**6)
thin_by = 10


if __name__ == "__main__":
    args = parse_args()
    seed = 0
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)

    true_mean = jax.random.normal(keys[1], shape=(dim,))
    true_precision = _make_covariance_skewed(keys[0], dim, condition_number)
    
    log_prob = make_gaussian(true_precision, true_mean)

    sampler = make_sampler(move_type=args.move,
                           total_chains=total_chains,
                           dim=dim,
                           log_prob=log_prob,
                           step_size=args.hamiltonian_step_size,
                           L=args.hamiltonian_L)

    initial_state = jax.random.normal(keys[1], shape=(total_chains, dim))
    samples = sampler.run_mcmc(
        key=keys[1],
        initial_state=initial_state,
        num_samples=num_samples,
        warmup=warmup,
        thin_by=thin_by,)
    

    print('Diagonstics:')
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
    
    print('Truth vs Empirical:')
    empirical_mean = jnp.mean(samples, axis=(0,1))
    empirical_cov = jnp.cov(samples.reshape(-1, dim).T)
    
    mean_residual = jnp.linalg.norm(true_mean - empirical_mean)
    cov_residual = jnp.linalg.norm(jnp.linalg.inv(true_precision) - empirical_cov)

    print(f"  Mean residual: {mean_residual}")
    print(f"  Covariance residual: {cov_residual}")
    
    
    
    