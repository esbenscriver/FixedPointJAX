# Fixed-point solver
FixedPointJAX is a simple implementation of a fixed-point iteration algorithm in JAX.

## Installation

### CPU-only (Linux/macOS/Windows)
```bash
pip install -U jax
```

### GPU (NVIDIA, CUDA 12)
```bash
pip install -U "jax[cuda12]"
```

### TPU (Google Cloud TPU VM)
```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Usage

```python
import jax.numpy as jnp
from jax import random

from FixedPointJAX import FixedPointRoot

# Define the logit probabilities
def my_logit(x, axis=0):
	nominator = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
	denominator = jnp.sum(nominator, axis=axis, keepdims=True)
	return nominator / denominator
	
# Define the function for the fixed-point iteration
def my_fxp(x,s0):
	s = my_logit(x)
	z = jnp.log(s0 / s)
	return x + z, z
print('-----------------------------------------')
# Number of alternatives
J = 3

# Simulate probabilities
s0 = my_logit(random.uniform(key=random.PRNGKey(123), shape=(J, 1)))

# Set up fixed-point equation
fun = lambda x: my_fxp(x,s0)

# Initial guess
x0 = jnp.zeros_like(s0)

# Solve the fixed-point equation
x, (step_norm, root_norm, iterations) = FixedPointRoot(fun, x0)
print('-----------------------------------------')
print('----- Check solution to fixed-point -----')
print("x0:")
print(x0)
print("x:")
print(x)
print("fun(x):")
print(fun(x)[0])
print('------- Check logit probabilities -------')
print("s0:")
print(s0)
print("s(x):")
print(my_logit(x))
print('-----------------------------------------')
```