import hemcee
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", False)

from configuration import parse_args, make_sampler, plot_samples

# RNG Settings will be set from command line arguments

# Distribution Settings
dim = 12
n_components = int(dim//2)  # Number of Gaussian components
overlap_factor = 0.2  # Controls overlap between modes (0 = no overlap, 1 = high overlap)

# Sampler Settings
total_chains = dim * 2
warmup = int(2 * 10**5)
num_samples = int(10**6)
thin_by = 10


def make_gaussian_mixture(dim, n_components, overlap_factor, key):
    """
    Create a Gaussian mixture distribution with overlapping modes.
    
    Parameters:
    dim: int, dimension of the state space
    n_components: int, number of Gaussian components
    overlap_factor: float, controls overlap between modes (0-1)
    key: JAX random key
    
    
    Returns:
    log_prob: function that takes x and returns log probability
    true_means: array of true means for each component
    true_covs: array of true covariances for each component
    true_weights: array of true mixture weights
    """
    # Split the key for generating different components
    keys = jax.random.split(key, n_components + 1)
    
    # Generate means that are separated but with some overlap
    # Place means on a line with controlled separation
    mean_separation = 2.0 * (1.0 - overlap_factor)  # Larger separation for less overlap
    true_means = jnp.zeros((n_components, dim))
    
    for i in range(n_components):
        # Create a direction vector
        direction = jax.random.normal(keys[i], shape=(dim,))
        direction = direction / jnp.linalg.norm(direction)
        
        # Place mean at distance mean_separation * i from origin
        true_means = true_means.at[i].set(direction * mean_separation * i)
    
    # Generate covariances (all components have same shape but different orientations)
    # Use a base covariance matrix
    base_cov = jnp.eye(dim) * 0.5  # Base covariance
    
    # Add some rotation to make it more interesting
    rotation_angle = jnp.pi / 4  # 45 degrees
    rotation_matrix = jnp.array([
        [jnp.cos(rotation_angle), -jnp.sin(rotation_angle)],
        [jnp.sin(rotation_angle), jnp.cos(rotation_angle)]
    ])
    
    # Extend rotation to full dimension (only rotate first 2 dimensions)
    full_rotation = jnp.eye(dim)
    full_rotation = full_rotation.at[:2, :2].set(rotation_matrix)
    
    true_covs = jnp.array([base_cov for _ in range(n_components)])
    true_covs = jnp.array([full_rotation @ cov @ full_rotation.T for cov in true_covs])
    
    # Generate mixture weights (equal weights for simplicity)
    true_weights = jnp.ones(n_components) / n_components
    
    def log_prob(x):
        """
        Compute log probability of Gaussian mixture.
        
        Args:
        x: Array of shape (dim,) representing a single sample
        
        Returns:
        Log probability of the mixture
        """
        # Vectorized computation for all components at once
        # Shape: (n_components, dim)
        diff = x[None, :] - true_means  # Broadcasting: (1, dim) - (n_components, dim)
        
        # Compute precision matrices for all components
        # Shape: (n_components, dim, dim)
        precisions = jnp.linalg.inv(true_covs)
        
        # Compute quadratic forms for all components
        # Shape: (n_components,)
        quadratic_forms = jnp.sum(diff[:, None, :] @ precisions @ diff[:, :, None], axis=(1, 2))
        
        # Compute log determinants for all components
        # Shape: (n_components,)
        log_dets = jnp.log(jnp.linalg.det(2 * jnp.pi * true_covs))
        
        # Compute log probabilities for all components
        # Shape: (n_components,)
        log_probs = -0.5 * (quadratic_forms + log_dets)
        
        # Add log weights
        log_weights = jnp.log(true_weights)
        weighted_log_probs = log_probs + log_weights
        
        # Use log-sum-exp trick for numerical stability
        max_log_prob = jnp.max(weighted_log_probs)
        log_prob_mixture = max_log_prob + jnp.log(jnp.sum(jnp.exp(weighted_log_probs - max_log_prob)))
        
        return log_prob_mixture
    
    return log_prob, true_means, true_covs, true_weights


def create_initial_state(key, total_chains, dim, true_means, n_components):
    """
    Create initial state for the Gaussian mixture.
    Start with samples near each mode.
    """
    # Split chains among components
    chains_per_component = total_chains // n_components
    remaining_chains = total_chains % n_components
    
    # Create array of chain counts per component
    chain_counts = jnp.full(n_components, chains_per_component)
    chain_counts = chain_counts.at[:remaining_chains].add(1)
    
    # Generate all random keys at once
    keys = jax.random.split(key, total_chains + 1)
    
    # Create initial states for all components at once
    initial_states = []
    key_idx = 0
    
    for i in range(n_components):
        n_chains = chain_counts[i]
        if n_chains > 0:
            # Generate initial states near this component's mean
            noise = 0.1 * jax.random.normal(keys[key_idx], shape=(n_chains, dim))
            component_states = true_means[i] + noise
            initial_states.append(component_states)
            key_idx += n_chains
    
    # Combine all initial states
    initial_state = jnp.vstack(initial_states)
    
    return initial_state


if __name__ == "__main__":
    import time
    
    args = parse_args()
    
    print("="*60)
    print("GAUSSIAN MIXTURE MCMC SAMPLING")
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
    print(f"Number of components: {n_components}")
    print(f"Overlap factor: {overlap_factor}")
    print(f"Total chains: {total_chains}")
    print(f"Warmup samples: {warmup}")
    print(f"Main samples: {num_samples}")
    print(f"Thin by: {thin_by}")
    print(f"Seed: {args.seed}")
    print("="*60)
    
    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, 3)

    print("Generating Gaussian mixture parameters...")
    log_prob, true_means, true_covs, true_weights = make_gaussian_mixture(
        dim, n_components, overlap_factor, keys[0]
    )
    print("✓ Gaussian mixture parameters generated")
    print(f"  True means: {true_means}")
    print(f"  True weights: {true_weights}")

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

    print("Initializing chains near mixture modes...")
    initial_state = create_initial_state(keys[1], total_chains, dim, true_means, n_components)
    print("✓ Chains initialized near mixture modes")
    
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
    
    print('Gaussian Mixture Analysis:')
    # Flatten samples for analysis
    flat_samples = samples.reshape(-1, dim)
    
    # Compute empirical mean and covariance
    empirical_mean = jnp.mean(flat_samples, axis=0)
    empirical_cov = jnp.cov(flat_samples.T)
    
    print(f"  Empirical mean: {empirical_mean}")
    print(f"  Empirical mean magnitude: {jnp.linalg.norm(empirical_mean):.3f}")
    print(f"  Empirical covariance trace: {jnp.trace(empirical_cov):.3f}")
    
    # Analyze mode detection
    print("  Mode Analysis:")
    for i in range(n_components):
        # Compute distance from each sample to each true mean
        distances = jnp.linalg.norm(flat_samples - true_means[i], axis=1)
        
        # Find samples close to this mode (within 2 standard deviations)
        mode_threshold = 2.0 * jnp.sqrt(jnp.trace(true_covs[i]))
        samples_near_mode = distances < mode_threshold
        mode_fraction = jnp.mean(samples_near_mode)
        
        print(f"    Mode {i+1}: {mode_fraction:.3f} fraction of samples near true mean")
        print(f"    Mode {i+1} true mean: {true_means[i]}")
        print(f"    Mode {i+1} mean distance: {jnp.mean(distances):.3f}")
    
    # Compute mixture weight estimates
    print("  Mixture Weight Analysis:")
    # Vectorized mode assignment: compute all distances at once
    # Shape: (n_samples, n_components)
    distances = jnp.linalg.norm(flat_samples[:, None, :] - true_means[None, :, :], axis=2)
    mode_assignments = jnp.argmin(distances, axis=1)
    
    for i in range(n_components):
        empirical_weight = jnp.mean(mode_assignments == i)
        print(f"    Mode {i+1}: empirical weight = {empirical_weight:.3f}, true weight = {true_weights[i]:.3f}")
    
    # Compute log probability statistics (vectorized)
    log_probs = jax.vmap(log_prob)(flat_samples)
    print(f"  Log probability statistics:")
    print(f"    Mean log prob: {jnp.mean(log_probs):.3f}")
    print(f"    Std log prob: {jnp.std(log_probs):.3f}")
    print(f"    Min log prob: {jnp.min(log_probs):.3f}")
    print(f"    Max log prob: {jnp.max(log_probs):.3f}")
    
    # Generate corner plot if requested
    if args.plot:
        title = f"Gaussian Mixture - {args.move}"
        filename = f"gaussian_mixture_samples_corner_{args.move}_s{args.hamiltonian_step_size}_L{args.hamiltonian_L}"
        plot_samples(samples, dim, title, filename)
