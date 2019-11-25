# %% codecell
# 1.2.2 Example: Librarian or Farmer?
%matplotlib inline
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt

figsize(12.5, 4)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

colors = ["#348ABD", "#A60628"]
prior = [1/21., 20/21.]
posterior = [.087, 1-.087]
plt.bar([0, .7], prior, alpha = .7, width = .25, color = colors[0],
        label = "prior distribution", lw="3", edgecolor = "#348ABD")

plt.bar([0+0.25, .7+0.25], posterior, alpha=0.7,
        width=0.25, color=colors[1],
        label="posterior distribution",
        lw="3", edgecolor="#A60628")

plt.xticks([0.20, 0.95], ["Librarian", "Farmer"])
plt.title("Prior and posterior probabilities of Steve's\
          occupation")
plt.ylabel("Probability")
plt.legend(loc="upper left")

# %% codecell
# 1.3.1 Discrete Case

figsize(12.5, 4)

import scipy.stats as stats
a = np.arange(16)
poi = stats.poisson
lambda_ = [1.5, 4.25]
colors = ["#348ABD", "#A60628"]

plt.bar(a, poi.pmf(a, lambda_[0]), color = colors[0],
        label = "$\lambda = %.1f$" % lambda_[0], alpha = .6,
        edgecolor=colors[0], lw="3")

plt.bar(a, poi.pmf(a, lambda_[1]), color=colors[1],
        label="$\lambda = %.1f$" % lambda_[1], alpha=.6,
        edgecolor =colors[1], lw ="3")

plt.xticks(a + .4, a)
plt.legend()
plt.ylabel("Probability of $k$")
plt.xlabel("$k$")
plt.title("Probability mass function of a Poisson random variable, \
            differeing \$\lambda$ values")

# %% codecell
# 1.3.2 Continuous Case

a = np.linspace(0, 4, 100)
expo = stats.expon
lambda_ = [.5, 1]

for l, c in zip(lambda_, colors):
    plt.plot(a, expo.pdf(a, scale=1./l), lw=3,
            color = c, label = "$\lambda = %.1f$" % l)
    plt.fill_between(a, expo.pdf(a, scale=1./l), color = c, alpha=.33)

plt.legend()
plt.ylabel("Probability density function at $z$")
plt.xlabel("$z$")
plt.ylim(0, 1.2)
plt.title("Probability density function of an exponential random\
            variable, differeing $\lambda$ values")

# %% codecell
# 1.4.1 Example: Inferring Behavior from Text-Message Data

figsize(12.5, 3.5)
count_data = np.loadtxt("txtdata.csv")
n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data, color = "#348ABD")
plt.xlabel("Time (days)")
plt.ylabel("Text messages received")
plt.title("Did the user's texting habits change over time?")
plt.xlim(0, n_count_data)

# %% codecell
# 1.4.2 Introducing our First Hammer: PyMc

import pymc3 as pm
import theano.tensor as tt

with pm.Model() as model:
    alpha = 1.0/count_data.mean()

    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)

    tau = pm.DiscreteUniform("tau", lower = 0, upper=n_count_data - 1)

with model:
    idx = np.arange(n_count_data)
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)

with model:
    observation = pm.Poisson("obs", lambda_, observed=count_data)

### Mysterious code to be explained in ch 3
with model:
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000, step=step)

lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
tau_samples = trace['tau']

figsize(12.5, 10)
#histogram of the samples

ax = plt.subplot(311)
ax.set_autoscaley_on(False)

plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=.85,
        label="posterior of $\lambda_1$", color="#A60628", density=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distribution of the variables $\lambda_1,\;\lambda_2,\;tau$""")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")

ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=.85,
        label="posterior of $\lambda_2$$", color="#7A68A6", density = True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")

plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins= n_count_data, alpha = 1,
        label=r"posterior of $\tau$",
        color="#467821", weights = w, rwidth = 2.)
plt.xticks(np.arange(n_count_data))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data)-20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("probability")

# %% codecell
# 1.4.4 What good are samples of the posterior anyways?

figsize(12.5, 5)
# tau_samples, lambda_1_samples, lambda_2_samples contain
# N samples form the corresponding posterior distribution
N = tau_samples.shape[0]
expected_texts_per_day = np.zeros(n_count_data)
for day in range(0, n_count_data):
    # ix is a bool index of all tau samples corresponding to
    # the switchpoint occuring prior to value of 'day'
    ix = day < tau_samples
    # Each posterior sample corresponds to a value for tau.
    # for each day, that value of tau indicated whether we're before
    # (in the lambda1 regime) or after in the lambda2 regime the switchpoint
    # by taking the posterior sample of lambda1/2 accordingly, we can average 
    # over all samples to get an expected value for lambda on that day.
    # As explained, the "message count" random variable is Poisson distributed,
    # and therefore lambda (the poisson parameter) is the expected value of
    # "message count".
    expected_texts_per_day[day] = (lambda_1_samples[ix].sum()
                                    + lambda_2_samples[~ix].sum()) / N

plt.plot(range(n_count_data), expected_texts_per_day, lw=4, color="#E24A33",
        label="expected number of text-messages received")
plt.xlim(0, n_count_data)
plt.xlabel("Day")
plt.ylabel("Expected # text-messages")
plt.title("Expected number of text-messages received")
plt.ylim(0, 60)
plt.bar(np.arange(len(count_data)), count_data, color="#348ABD", alpha=.65,
        label="observed texts per day")
plt.legend(loc="upper left")

# %% codecell
# What is the the mean posterior distribution for lambda_1 & lambda_2?
print(lambda_1_samples.mean())
print(lambda_2_samples.mean())
# What is the expected percentage increase in text-message rates?
 (lambda_2_samples-lambda_1_samples)/lambda_1_samples
 # 3
 ix = tau_samples < 45
 print(lambda_1_samples[ix].mean())
