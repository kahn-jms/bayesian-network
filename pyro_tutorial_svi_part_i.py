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

# # SVI Part I
#
# http://pyro.ai/examples/svi_part_i.html

# ## Setup

# The different pieces of model() are encoded via the mapping:
#
# * observations: `pyro.sample` with the `obs` argument
# * latent random variables: `pyro.sample`
# * parameters: `pyro.param`
#
# Note that the `obs` is used for conditioning, either you call `pyro.condition()` with on the model and use the `data` parameter to pass a dictionary of `{'sample name': observation}`, or you can call `obs=value` directly inside `pyro.sample`. 
#
# As a gotcha: In Pyro they use "latent variables" ($\mathbf{z}$) to mean unobservable `pyro.sample` calls, not the latent variables from BNN+LV.

# ## Model learning

# Note that the last equation shown is just Bayes theorem.

# ## Guide

# Want to know the posterior distribution $p(\mathbf{z}|\mathbf{x})$ but it's intractable.
# Instead approximate it with the **guide** $q_{\phi}(\mathbf{z})$, where $\phi$ are the variational parameters.
#
# The guide does not contain observations as it must be a properly normalised distribution.
#
# For every latent variable (`pyro.sample` statement without an observation) in the `model()`, there must be a matching `pyro.sample` in the `guide()` with the same name (e.g. `pyro.sample('z_1')`).
# The actual distributions called in each can be different, just the names must match.

# ## Evidence Lower Bound (ELBO)

# Need to revisit this and check how ELBO gets to KL divergence.

# ## `SVI` class

# `SVI` does variational inference in Pyro.
# Currently only supports ELBO loss, in future more will be added.
#
# The user needs to provide three things: the model, the guide, and an optimizer.
# As an example:
# ```python
# import pyro
# from pyro.infer import SVI, Trace_ELBO
# svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
# ```

# ## Optimizers

# Pyro provides its own optimizers you have to use.
# These are just wrappers around PyTorch optimizers.
# The difference is that Pyro will generate a new optimizer for every parameter in the model (every `pyro.param`).
# This is because the `model()` and the `guide()` parameters can be created dynamically during learning (since they are named within the functions, the names can be dynamic so long as they match between the model and the guide).
#
# For most cases we don't care much and just call something like:
# ```python
# from pyro.optim import Adam
#
# adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
# optimizer = Adam(adam_params)
# ```
# which will apply the Adam optimizer with identical settings to all parameters.
#
# Othersie you can pass a callable to the optimizer with the arguments `module_name` and `param_name`, e.g.:
# ```python
# from pyro.optim import Adam
#
# def per_param_callable(module_name, param_name):
#     if param_name == 'my_special_parameter':
#         return {"lr": 0.010}
#     else:
#         return {"lr": 0.001}
#
# optimizer = Adam(per_param_callable)
# ```

# ## A simple example

import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 2000

# enable validation (e.g. validate parameters of distributions)
# We have a newer version
# assert pyro.__version__.startswith('1.3.0')
pyro.enable_validation(True)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

# create some data with 6 observed heads and 4 observed tails
data = []
for _ in range(6):
    data.append(torch.tensor(1.0))
for _ in range(4):
    data.append(torch.tensor(0.0))


def model(data):
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    # sample f from the beta prior
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
    # loop over the observed data
    for i in range(len(data)):
        # observe datapoint i using the bernoulli likelihood
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])


def guide(data):
    # register the two variational parameters with Pyro
    # - both parameters will have initial value 15.0.
    # - because we invoke constraints.positive, the optimizer
    # will take gradients on the unconstrained parameters
    # (which are related to the constrained parameters by a log)
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0),
                         constraint=constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.0),
                        constraint=constraints.positive)
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))


# setup the optimizer
adam_params = {"lr": 0.001, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# do gradient steps
for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print('.', end='')

# grab the learned variational parameters
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()

alpha_q, beta_q

# here we use some facts about the beta distribution
# compute the inferred mean of the coin's fairness
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * math.sqrt(factor)

print("\nbased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))
