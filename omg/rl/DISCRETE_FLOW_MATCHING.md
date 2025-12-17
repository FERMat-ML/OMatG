# Discrete Flow Matching with Residual Learning

## The Challenge

Your base model uses **discrete flow matching** for atomic species, which works fundamentally differently from continuous flow matching:

- **Continuous (positions, cell)**: Velocity fields `v(x,t)`
- **Discrete (species)**: Rate matrices `R(t)` over categorical distributions

## Solution: Residual Rate Matrices

For discrete species, the residual model should learn **rate matrix corrections** instead of velocity corrections.

### Conceptual Approach

**Base model:**
- Outputs: `p_base(species | x, t)` - probability distribution over atomic species
- Governed by rate matrix `R_base`

**Residual model:**
- Outputs: `logits_residual(x, t)` - corrections to the rate matrix
- Combined: `p_total = combine(p_base, p_residual)`

**During RL training:**
- Add stochastic noise via **Gumbel-Softmax** (continuous relaxation of discrete)
- Use **detailed balance noise** during integration (as in your `DiscreteFlowMatchingMask`)

## Implementation

### 1. ResidualRateMatrix Class

```python
from omg.rl.residual_model_discrete import ResidualRateMatrix

# Create residual model for discrete species
discrete_residual = ResidualRateMatrix(
    base_architecture=your_model,
    noise_scale=1.0,  # Temperature for Gumbel-Softmax
    regularization_weight=0.01,
    mask_index=0,
)
```

### 2. Key Differences from Continuous

| Aspect | Continuous (pos, cell) | Discrete (species) |
|--------|------------------------|---------------------|
| **Output** | Velocity field | Logits over species |
| **Noise** | Gaussian | Gumbel-Softmax |
| **Combination** | `v_total = v_base + v_residual` | `p_total = combine(p_base, p_residual)` |
| **Integration** | Euler/RK steps | Categorical sampling + detailed balance |
| **Regularization** | L2 on velocities | KL divergence from uniform |

### 3. How It Works

**Training:**
1. Base model outputs: `p_base(species)`
2. Residual outputs: `logits_residual` (with Gumbel noise)
3. Combine to get: `p_total`
4. Sample species from `p_total`
5. Compute reward
6. Update residual via policy gradients

**Inference:**
1. Base model outputs: `p_base(species)`
2. Residual outputs: `logits_residual` (deterministic, no noise)
3. Combine to get: `p_total`
4. Sample species from `p_total` during integration
5. Use detailed balance noise (as in your base model)

### 4. Integration with Detailed Balance

The `integrate_with_detailed_balance()` method combines:

- **Base + residual probabilities**: Combined distribution over species
- **Detailed balance dynamics**: Masking/unmasking as in your `DiscreteFlowMatchingMask`

```python
x_t_new = discrete_residual.integrate_with_detailed_balance(
    base_probs=base_output,
    residual_logits=residual_output,
    x_t=current_species,
    time=t,
    time_step=dt,
    noise_param=0.0,  # Your noise parameter from DiscreteFlowMatchingMask
)
```

This maintains the detailed balance property while incorporating residual corrections.

## Rate Matrix Combination Strategies

Three ways to combine base and residual rate matrices:

### 1. Weighted Sum (Default)
```python
p_total = (1 - α) * p_base + α * p_residual
```
- Simple linear interpolation
- α controls residual influence
- Easy to interpret

### 2. Product of Experts
```python
p_total ∝ p_base * p_residual
```
- Both models must agree
- Good for conservative corrections
- Normalized to sum to 1

### 3. Geometric Mean
```python
p_total ∝ sqrt(p_base * p_residual)
```
- Compromise between base and residual
- Symmetric influence

## Gumbel-Softmax for Policy Gradients

**Why Gumbel-Softmax?**
- Discrete sampling is not differentiable
- Gumbel-Softmax provides continuous relaxation
- Allows gradients to flow during training
- Temperature controls "discreteness"

**During training:**
```python
# Add Gumbel noise
gumbel_noise = -log(-log(uniform(0,1)))
noisy_logits = (logits + gumbel_noise) / temperature

# Sample (approximately)
probs = softmax(noisy_logits)
```

**During inference:**
```python
# No noise, just use mean
probs = softmax(logits)
```

## Regularization

For discrete residuals, we use **KL divergence from uniform**:

```python
# Encourage residual to make small corrections
# (stay close to uniform distribution)
probs = softmax(residual_logits)
uniform = 1 / MAX_ATOM_NUM

reg_loss = KL(probs || uniform)
```

This keeps rate matrix corrections small, analogous to L2 regularization for continuous case.

## Example Usage

### Basic Setup

```python
from omg.rl.residual_model import ResidualModel  # For continuous
from omg.rl.residual_model_discrete import ResidualRateMatrix  # For discrete

# Continuous residual (positions, cell)
continuous_residual = ResidualModel(
    base_architecture=model_for_continuous,
    noise_scale=0.1,
    regularization_weight=0.01,
)

# Discrete residual (species)
discrete_residual = ResidualRateMatrix(
    base_architecture=model_for_discrete,
    noise_scale=1.0,  # Temperature
    regularization_weight=0.01,
)
```

### Hybrid RL Training

For a full system with both continuous and discrete:

```python
class HybridResidualModel(nn.Module):
    """Combines continuous and discrete residuals."""

    def __init__(self, continuous_model, discrete_model):
        super().__init__()
        self.continuous = ResidualModel(continuous_model, ...)
        self.discrete = ResidualRateMatrix(discrete_model, ...)

    def forward(self, x, t, return_log_prob=True):
        # Get continuous residuals (pos, cell)
        cont_output, cont_log_prob, cont_mean = self.continuous(x, t, return_log_prob)

        # Get discrete residuals (species)
        disc_output, disc_log_prob, disc_mean = self.discrete(x, t, return_log_prob)

        # Combine outputs
        output = {**cont_output, **disc_output}

        # Combine log probs
        if return_log_prob and cont_log_prob is not None and disc_log_prob is not None:
            total_log_prob = cont_log_prob + disc_log_prob.sum(dim=-1)
        else:
            total_log_prob = None

        mean = {**cont_mean, **disc_mean}

        return output, total_log_prob, mean
```

## Integration During Generation

During generation, you need to handle discrete species specially:

```python
# Get base outputs
base_output = base_model(x_t, t)
base_species_probs = F.softmax(base_output['species_b'], dim=-1)

# Get residual outputs
residual_output, _, _ = discrete_residual(x_t, t, return_log_prob=False)
residual_logits = residual_output['species_b']

# Combine
combined_probs = combine_rate_matrices(
    base_species_probs,
    residual_logits,
    combination_mode='weighted_sum',
    alpha=0.1,
)

# Integrate with detailed balance
x_t['species'] = discrete_residual.integrate_with_detailed_balance(
    base_probs=combined_probs,
    residual_logits=residual_logits,
    x_t=x_t['species'],
    time=t,
    time_step=dt,
    noise_param=base_model.noise,  # From DiscreteFlowMatchingMask
)
```

## Technical Details

### Detailed Balance Preservation

The detailed balance property from your `DiscreteFlowMatchingMask` is preserved because:

1. **Unmasking rate**: Proportional to combined probability distribution
2. **Masking rate**: Controlled by noise parameter (same as base)
3. **Residual only affects target distribution**, not the balance dynamics

### Temperature Annealing

Like continuous noise, you can anneal Gumbel-Softmax temperature:

```python
# Start with high temperature (more exploration)
initial_temp = 2.0

# Anneal to low temperature (more deterministic)
final_temp = 0.1

# During training
current_temp = initial_temp * (decay_factor ** iteration)
discrete_residual.set_noise_scale(current_temp)
```

## Comparison with Base Model

| Component | Base (DiscreteFlowMatchingMask) | With Residual |
|-----------|----------------------------------|---------------|
| **Output** | `p(species \| x, t)` | `p_base + p_residual` |
| **Training** | Cross-entropy loss | RL (reward-based) |
| **Noise** | Detailed balance | Detailed balance + Gumbel |
| **Goal** | Match training distribution | Maximize reward |

## Future Enhancements

1. **Learned combination weight α**: Instead of fixed, learn α(x, t)
2. **Continuous-time Markov process**: As noted in your TODO
3. **More sophisticated rate matrix combinations**: e.g., attention-based
4. **Conditional residuals**: Condition on structure properties
5. **Multi-scale residuals**: Different residuals for different time scales

## References

- Your `DiscreteFlowMatchingMask` implementation
- Discrete Flow Matching paper: https://arxiv.org/pdf/2402.04997
- Gumbel-Softmax: https://arxiv.org/abs/1611.01144
- Detailed balance in discrete flows

## Summary

**Key insight:** Discrete flow matching requires rate matrix residuals, not velocity residuals.

**Implementation:**
- Use `ResidualRateMatrix` for species
- Use `ResidualModel` for positions/cell
- Combine both in hybrid model
- Integrate with detailed balance

**Benefits:**
- Respects discrete nature of species
- Maintains detailed balance
- Allows RL optimization
- Consistent with your base model

The discrete residual framework extends your existing discrete flow matching naturally!
