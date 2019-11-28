import pymc3 as pm

with pm.Model() as model:
    parameter = pm.Exponential("poisson_param", 1.0)
    data_generator = pm.Poisson("data_generator", parameter)

with model:
    data_plus_one = data_generator + 1

parameter.tag.test_value

with pm.Model() as model:
    theta = pm.Exponential("theta", 2.0)
    data_generator = pm.Poisson("data_generator", theta)

with pm.Model() as ab_testing:
    p_A = pm.Uniform("P(A)", 0, 1)
    p_B = pm.Uniform("P(B)", 0, 1)

# %% codecell
# pymc3 variables

print("parameter.tag.test_value =", parameter.tag.test_value)
print("data_generator.tag.test_value =", data_generator.tag.test_value)
print("data_plus_one.tag.test_value =", data_plus_one.tag.test_value)

with pm.Model() as model:
    parameter = pm.Exponential("poisson_param", 1.0, testval=.5)

print("\nparameter.tag.test_value =", parameter.tag.test_value)

# Determanistic variables
with pm.Model() as model:
    lambda_1 = pm.Exponential("lambda_1", 1.0)
    lambda_2 = pm.Exponential("lambda_2", 1.0)
    tau = pm.DiscreteUniform("tau", lower=0, upper = 10)

new_deterministic_variable = lambda_1+lambda_2

import numpy as np

n_data_points = 5
idx = np.arange(n_data_points)
with model:
    lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)

# %% codecell
# Theano
import theano.tensor as tt

with pm.Model() as theano_test:
    p1 = pm.Uniform("p", 0, 1)
    p2 = 1 - p1
    p = tt.stack([p1, p2])

    assignment = pm.Categorical("assignment", p)

# %% codecell
# Including obs in the model

%matplotlib inline
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import scipy.stats as stats
figsize(12.5, 4)

samples = lambda_1.random(size=20000)
plt.hist(samples, bins=70, density=True, histtype="stepfilled")
plt.title("Prior distribution for $\lambda_1$")
plt.xlim(0, 8)

data = np.array([10, 5])
with model:
    fixed_variable = pm.Poisson("fxd", 1, observed=data)
print("value: ", fixed_variable.tag.test_value)

# We're using some fake data here
data = np.array([10, 25, 15, 20, 35])
with model:
    obs = pm.Poisson("obs", lambda_, observed = data)
print(obs.tag.test_value)

# %% codecell
#Same story, different ending
tau = np.random.randint(0,80)
print(tau)

alpha = 1./20.
lambda_1, lambda_2 = np.random.exponential(scale=1/alpha, size=2)
print(lambda_1, lambda_2)

data = np.r_[stats.poisson.rvs(mu=lambda_1, size=tau), stats.poisson.rvs(mu=lambda_2, size=80-tau)]

plt.bar(np.arange(80), data, color="#348ABD")
plt.bar(tau-1, data[tau - 1], color="r", label="user behaviour changed")
plt.xlabel("Time (days)")
plt.ylabel("count of text-msgs received")
plt.title("Artificial dataset")
plt.xlim(0, 80)
plt.legend()


def plot_artificial_sms_dataset():
    tau = stats.randint.rvs(0, 80)
    alpha = 1./20.
    lambda_1, lambda_2 = stats.expon.rvs(scale=1/alpha, size=2)
    data = np.r_[stats.poisson.rvs(mu=lambda_1, size=tau), stats.poisson.rvs(mu=lambda_2, size=80 - tau)]
    plt.bar(np.arange(80), data, color="#348ABD")
    plt.bar(tau - 1, data[tau-1], color="r", label="user behaviour changed")
    plt.xlim(0, 80);

figsize(12.5, 5)
plt.title("More example of artificial datasets")
for i in range(4):
    plt.subplot(4, 1, i+1)
    plot_artificial_sms_dataset()

# %%
# Bayesian A/B Testing
import pymc3 as pm

# The parameters are the bounds of the Uniform.
with pm.Model() as model:
    p = pm.Uniform('p', lower = 0, upper = 1)

# set constants
p_true = 0.05 # remember, this is unknown
N = 1500

# sample N Bernoulli random variables from Ber(0.05).
# each random variable has a 0.05 chance of being 1.
# this is the data-generation step
occurances = stats.bernoulli.rvs(p_true, size = N)

print(occurances)
print(np.sum(occurances))

# Occurances.mean is equal to n/N
print("What is the observed frequency in Group A? %.4f" % np.mean(occurances))
print("Does this equal the true frequency? %s" % (np.mean(occurances) == p_true))

# include the observations, which are Bernoulli
with model:
    obs = pm.Bernoulli("obs", p, observed=occurances)
    # To be explained in ch 3
    step = pm.Metropolis()
    trace = pm.sample(18000, step = step)
    burned_trace = trace[1000:]

figsize(12.5, 4)
plt.title("Posterior distribution of $p_A$, the true effectiveness of Site A")
plt.vlines(p_true, 0, 90, linestyle="--", label="true $p_A (unknown)")
plt.hist(burned_trace["p"], bins = 25, histtype="stepfilled", normed=True)
plt.legend()

# A and B together
import pymc3 as pm
figsize(12,4)

# these two quantities are unknown to us.
true_p_A = .05
true_p_B = .04

# notice the unequal sample sizes -- no problem in Bayesian analysis.
N_A = 1500
N_B = 750

# generate some observations
observations_A = stats.bernoulli.rvs(true_p_A, size = N_A)
observations_B = stats.bernoulli.rvs(true_p_B, size = N_B)
print("Obs from Site A: ", observations_A[:30], "...")
print("Obs from Site B: ", observations_B[:30], "...")
print(np.mean(observations_A))
print(np.mean(observations_B))

# Set up the pymc3 model. Again assume Uniform priors for p_A and p_B.
with pm.Model() as model:
    p_A = pm.Uniform("p_A", 0, 1)
    p_B = pm.Uniform("p_B", 0, 1)

    # Define the deterministic delta function. This is our unknown of interest.
    delta = pm.Deterministic("delta", p_A - p_B)

    # Set of observations, in this case we have two observation datasets.
    obs_A = pm.Bernoulli("obs_A", p_A, observed = observations_A)
    obs_B = pm.Bernoulli("obs_B", p_B, observed = observations_B)

    # To be explained in ch 3.
    step = pm.Metropolis()
    trace = pm.sample(20000, step = step)
    burned_trace=trace[1000:]

p_A_samples = burned_trace["p_A"]
p_B_samples = burned_trace["p_B"]
delta_samples = burned_trace["delta"]

figsize(12.5, 10)

# histogram of posteriors

ax = plt.subplot(311)

plt.xlim(0, .1)
plt.hist(p_A_samples, histtype='stepfilled', bins = 25, alpha = .85,
        label = "posterior of $p_A$", color = "#A60628", normed=True)
plt.vlines(true_p_A, 0, 80, linestyle = "--", label = "true $p_A (unknown)")
plt.legend(loc="upper right")
plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")

ax = plt.subplot(312)

plt.xlim(0, .1)
plt.hist(p_B_samples, histtype='stepfilled', bins=25, alpha =.85,
        label="posterior of $p_B$", color="#467821", normed=True)
plt.vlines(true_p_B, 0, 80, linestyle="--", label="true $p_B$ (unkown)")
plt.legend(loc="upper right")

ax = plt.subplot(313)
plt.hist(delta_samples, histtype='stepfilled', bins=30, alpha=.85,
        label="posterior of delta", color="#7A68A6", normed=True)
plt.vlines(true_p_A - true_p_B, 0, 60, linestyle="--",
            label="true delta (unknown)")
plt.vlines(0, 0, 60, color="black", alpha=.2)
plt.legend(loc="upper right")

# Count the number of samples less than 0, i.e. the area under the curve
# before 0, represent the probability that site A is worse than B
print("Probability site A is WORSE than site B: %.3f" % \
    np.mean(delta_samples < 0))
print("Probability site A is BETTER than site B: %.3f" % \
    np.mean(delta_samples > 0))

# %% codecell
# An algorithm for human deceitt
figsize(12.5, 4)

import scipy.stats as stats
binomial = stats.binom

parameters = [(10, .4), (10, .9)]
colors =["#348ABD", "#A60628"]

for i in range(2):
    N, p = parameters[i]
    _x = np.arange(N + 1)
    plt.bar(_x -.5, binomial.pmf(_x, N, p), color=colors[i],
        edgecolor=colors[i],
        alpha=.6,
        label="$N$: %d, $p$: %.1f" % (N, p),
        linewidth=3)

plt.legend(loc="upper left")
plt.xlim(0, 10.5)
plt.xlabel("$k$")
plt.ylabel("$P(X = k)$")
plt.title("Probability mass distributions of binomial random variables")

# cheating students
import pymc3 as pm

N = 100
with pm.Model() as model:
    p = pm.Uniform("freq_cheating", 0, 1)

with model:
    true_answers = pm.Bernoulli("truths", p, shape = N, testval = np.random.binomial(1, .5, N))
