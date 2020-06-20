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

# # Pyro playground
#
# Initial place to play with Pyro for learning stochastic functions.

# +
import torch
import pyro
import matplotlib.pyplot as plt

# Set matplotlib settings
# %matplotlib inline
plt.style.use('default')
# -

pyro.set_rng_seed(101)

# ## Primitive Stochastic Functions

loc = 0.   # mean zero
scale = 1. # unit variance
normal = torch.distributions.Normal(loc, scale) # create a normal distribution object
x = normal.rsample() # draw a sample from N(0,1)
print("sample", x)
print("log prob", normal.log_prob(x)) # score the sample from N(0,1)

# +
plt.figure()

plt.hist(
    normal.rsample((10000,)),
    bins='auto',
)
plt.show()


# -

# ## A Simple Model

def weather():
    cloudy = torch.distributions.Bernoulli(0.3).sample()
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    temp = torch.distributions.Normal(mean_temp, scale_temp).rsample()
    return cloudy, temp.item()


weather()

# ## The `pyro.sample` Primitive

x = pyro.sample("my_sample", pyro.distributions.Normal(loc, scale))
print(x)


def weather():
    cloudy = pyro.sample('cloudy', pyro.distributions.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()


for _ in range(3):
    print(weather())


# ## Universality: Stochastic Recursion, Higher-order Stochastic Functions, and Random Control Flow

def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = 200. if cloudy == 'sunny' and temp > 80.0 else 50.
    ice_cream = pyro.sample('ice_cream', pyro.distributions.Normal(expected_sales, 10.0))
    return ice_cream, cloudy


ice_cream_sales()


# ### Recursion

def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t), pyro.distributions.Bernoulli(p))
    if x.item() == 1:
        return 0
    else:
        return 1 + geometric(p, t + 1)


print(geometric(0.01))

a = torch.tensor([geometric(0.01) for _ in range(1000)])

# +
plt.figure()

plt.hist(
    a,
    bins='auto',
)

plt.show()


# -

# ### Stochastic functions as input/output

def normal_product(loc, scale):
    z1 = pyro.sample("z1", pyro.distributions.Normal(loc, scale))
    z2 = pyro.sample("z2", pyro.distributions.Normal(loc, scale))
    y = z1 * z2
    return y


def make_normal_normal():
    mu_latent = pyro.sample("mu_latent", pyro.distributions.Normal(0, 1))
    fn = lambda scale: normal_product(mu_latent, scale)
    return fn


print(make_normal_normal()(1.))

t = torch.tensor([normal_product(0,1) for _ in range(1000)])

t = torch.tensor([make_normal_normal()(100.) for _ in range(1000)])

# +
plt.figure()

plt.hist(
    t,
    bins='auto',
)
plt.show()
