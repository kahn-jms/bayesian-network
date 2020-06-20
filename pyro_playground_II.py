# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Pyro intro part ii
#
# From http://pyro.ai/examples/intro_part_ii.html

# +
import matplotlib.pyplot as plt
import numpy as np
import torch

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(101)

# Set matplotlib settings
# %matplotlib inline
plt.style.use('default')


# -

# ## A Simple Example

def scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    return pyro.sample("measurement", dist.Normal(weight, 0.75))


# +
plt.figure()

plt.hist(
    [scale(10) for _ in range(5000)],
    bins='auto',
)
plt.show()
# -

# ## Conditioning
#
# Pyro lets you condition on a named `pyro.sample` stochastic distribution.

# In this case conditioning on ($\text{measurement}=9.5$) will make `scale` always return $9.5$ because that's an observation (real data) so we already know the true outcome.

conditioned_scale = pyro.condition(scale, data={"measurement": 9.5})

conditioned_scale(guess=3)

# We can condition on any named distribution.
# In this case an observation of the true weight means we are just sample from a Normal ditribution.

conditioned_scale = pyro.condition(scale, data={"weight": 5})

plt.figure()
plt.hist(
    [conditioned_scale(1) for _ in range(1000)],
    bins='auto',
)
plt.show()


# Also called as a function:

def deferred_conditioned_scale(measurement, guess):
    return pyro.condition(scale, data={"measurement": measurement})(guess)


deferred_conditioned_scale(10, 5)


# ### Using the `obs` keyword

def scale_obs(guess):  # equivalent to conditioned_scale above
    weight = pyro.sample("weight", dist.Normal(guess, 1.))
     # here we condition on measurement == 9.5
    return pyro.sample("measurement", dist.Normal(weight, 0.75), obs=9.5)


scale_obs(1)


# ## Flexible Approximate Inference With Guide Functions
#
# Guide functions must satisfy these two criteria to be valid approximations for a particular model: 1. all unobserved (i.e., not conditioned) sample statements that appear in the model appear in the guide. 2. the guide has the same input signature as the model (i.e., takes the same arguments)
#
# Guides are approximate posterior distributions.

# This next one the tutorial links to how it is calcuated analytically.
# The true posterior is $\mathcal{N}(9.14,0.6)$

def perfect_guide(guess):
    loc =(0.75**2 * guess + 9.5) / (1 + 0.75**2) # 9.14
    scale = np.sqrt(0.75**2/(1 + 0.75**2)) # 0.6
    return pyro.sample("weight", dist.Normal(loc, scale))


perfect_guide(9.14)


# ## Parametrized Stochastic Functions and Variational Inference

def intractable_scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    return pyro.sample("measurement", dist.Normal(torch.nn.functional.elu(weight), 0.75))


intractable_scale(9.5)

# ### Storing params
#
# With `pyro.param`, the parameter is stored and can be recalled later from the param store.

a = pyro.param("a", torch.tensor(1.))
print(a, pyro.param('a').item())


def scale_parametrized_guide(guess):
    a = pyro.param("a", torch.tensor(guess))
    b = pyro.param("b", torch.tensor(1.))
    return pyro.sample("weight", dist.Normal(a, torch.abs(b)))


scale_parametrized_guide(10)

# Instead of `torch.abs()` you can use `constraint`

# +
from torch.distributions import constraints

def scale_parametrized_guide_constrained(guess):
    a = pyro.param("a", torch.tensor(guess))
    b = pyro.param("b", torch.tensor(1.), constraint=constraints.positive)
    return pyro.sample("weight", dist.Normal(a, b))  # no more torch.abs


# -

scale_parametrized_guide_constrained(10)

# ### Quick example of stochastic variational inference (SVI)

# +
guess = 8.5

pyro.clear_param_store()
svi = pyro.infer.SVI(model=conditioned_scale,
                     guide=scale_parametrized_guide,
                     optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
                     loss=pyro.infer.Trace_ELBO())


losses, a,b  = [], [], []
num_steps = 2500
for t in range(num_steps):
    losses.append(svi.step(guess))
    a.append(pyro.param("a").item())
    b.append(pyro.param("b").item())

plt.figure()
plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss");
plt.show()
print('a = ',pyro.param("a").item())
print('b = ', pyro.param("b").item())

# +
plt.subplot(1,2,1)
plt.plot([0,num_steps],[9.14,9.14], 'k:')
plt.plot(a)
plt.ylabel('a')

plt.subplot(1,2,2)
plt.ylabel('b')
plt.plot([0,num_steps],[0.6,0.6], 'k:')
plt.plot(b)
plt.tight_layout()
# -


