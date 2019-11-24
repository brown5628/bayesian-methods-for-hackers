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
# 1.4.1 Example: Infewrring Behavior from Text-Message Data

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

import pymc as pm

alpha = 1.0/count_data.mean()  # Recall that count_data is the
                               # variable that holds our text counts.
lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential("lambda_2", alpha)

tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)
