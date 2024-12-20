[![PyPI version](https://badge.fury.io/py/FixedPointJAX.svg)](https://badge.fury.io/py/FixedPointJAX)
# Fixed-point solver
FixedPointJAX is a simple implementation of a fixed-point iteration algorithm for root finding in JAX. The implementation allow the user to solve the system of fixed point equations by standard fixed point iterations and the SQUAREM accelerator, see [Du and Varadhan (2020)](https://www.jstatsoft.org/article/view/v092i07).

* Strives to be minimal
* Has no dependencies other than JAX

## Installation

```bash
pip install FixedPointJAX
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
# Dimensions of system of fixed-point equations
shape = (3, 4)

# Simulate probabilities
s0 = my_logit(random.uniform(key=random.PRNGKey(123), shape=shape))

# Set up fixed-point equation
fun = lambda x: my_fxp(x,s0)

# Initial guess
x0 = jnp.zeros_like(s0)

# Solve the fixed-point equation
x, (step_norm, root_norm, iterations) = FixedPointRoot(fun, x0)
print('-----------------------------------------')
print(f'System of fixed-point equations is solved: {jnp.allclose(x,fun(x)[0])}.')
print(f'Probabilities are identical: {jnp.allclose(s0, my_logit(x))}.')
print('-----------------------------------------')
```
