%matplotlib inline
import numpy as np
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt

figsize(12.5, 5)

sample_size = 100000
expected_value = lambda_ = 4.5
poi = np.random.poisson
N_samples = range(1, sample_size, 100)

for k in range(3):
    samples = poi(lambda_, sample_size)

    partial_average = [samples[:i].mean() for i in N_samples]

    plt.plot(N_samples, partial_average, lw=1.5, label="average \ pf $n$ samples; seq. %d"%k)

plt.plot(N_samples, expected_value*np.ones_like(partial_average),
    ls = "--", label="true expected value", c ="k")

plt.ylim(4.35, 4.65)
plt.title("Convergence of the average of \n random variables to its \
expected_value")
plt.ylabel("average of $n$ samples")
plt.xlabel( "# of samples, $n$")
plt.legend()

figsize(12.5, 4)

N_Y = 250
N_array = np.arange(1000, 50000, 2500)
D_N_results = np.zeros(len(N_array))

lambda_ = 4.5
expected_value = lambda_

def D_N(n):
    """
    This function approx. D_n, the average variance using n samples.
    """
    Z = poi(lambda_, (n, N_Y))
    average_Z = Z.mean(axis=0)
    return np.sqrt(((average_Z - expected_value)**2).mean())

for i,n in enumerate(N_array):
    D_N_results[i] = D_N(n)

plt.xlabel("$N$")
plt.ylabel("expected squared-distance from true value")
plt.plot(N_array, D_N_results, lw=3,
            label="expected distance between\n\
expected value and \naverage of $N$ random variables.")
plt.plot(N_array, np.sqrt(expected_value)/np.sqrt(N_array), lw=2, ls="--",
        label = r"$\frac{\sqrt{\lambda}}{\sqrt{N}}$")
plt.legend()
plt.title("How 'fast' is the sample average converging?")

N = 10000
print( np.mean( [ np.random.exponential( 0.5 ) > 5 for i in range(N) ] ) )

# %% codecell
figsize( 12.5, 4)
std_height = 15
mean_height = 150

n_counties = 5000
pop_generator = np.random.randint
norm = np.random.normal

#generate some artificial population numbers
population = pop_generator(100, 1500, n_counties)

average_across_county = np.zeros(n_counties)
for i in range(n_counties):
    average_across_county[i] = norm(mean_height, 1./std_height, population[i]).mean()

i_min = np.argmin(average_across_county)
i_max = np.argmax(average_across_county)

plt.scatter(population, average_across_county, alpha= .5, c="#7A68A6")
plt.scatter([population[i_min], population[i_max]],
            [average_across_county[i_min], average_across_county[i_max]],
            s = 60, marker = "o", facecolors = "none",
            edgecolors= "#A60628", linewidths = 1.5,
            label = "extreme heights")

plt.xlim(100, 1500)
plt.title("Average height vs. County Population")
plt.xlabel("County Population")
plt.ylabel("Average height in country")
plt.plot([100, 1500], [150, 150], color="k", label = "true expected \ height", ls ="--")
plt.legend(scatterpoints=1)


print("Population sizes of 10 'shortest' counties: ")
print(population[ np.argsort( average_across_county )[:10] ], '\n')
print("Population sizes of 10 'tallest' counties: ")
print(population[ np.argsort( -average_across_county )[:10] ])

# %% codecell
figsize(12.5, 6.5)
data = np.genfromtxt("./census_data.csv", skip_header=1, delimiter = ",")
plt.scatter(data[:,1], data[:,0], alpha=.5, c="#7A68A6")
plt.title("Census mail-back rate vs Population")
plt.ylabel("Mail-back rate")
plt.xlabel("population of block-group")
plt.xlim(-100, 15e3)
plt.ylim(-5, 105)

i_min = np.argmin(data[:,0])
i_max = np.argmax(data[:,0])

plt.scatter([data[i_min,1], data[i_max,1]],
            [data[i_min,0], data[i_max,0]],
            s = 60, marker="o", facecolors ="none",
            edgecolors = "#A60629", linewidths =1.5,
            label="most extreme points")

plt.legend(scatterpoints = 1)

# %% codecell
import sys
from IPython.core.display import Image
import praw

reddit = praw.Reddit("BayesianMethodsForHackers")
subreddit = reddit.get_subreddit("showerthoughts")

top_submissions =  subreddit.get_top(limit=100)

n_sub = int(sys.argv[1]) if sys.argv[1] else 1

i = 0
while i < n_sub:
    top_submission = next(top_submissions)
    i+=1

top_post = top_submission.title

upvotes = []
downvotes = []
contents = []

for sub in top_submissions:
    try:
        ratio = reddit.get_submissions(sub.permalink).upvote_ratio
        ups = int(round((ratio*sub.score)/(2*ratio-1)) if ratio !=.5 else round(sub.score/2))
        upvotes.append(ups)
        downvotes.append(ups-sub.score)
        contents.append(sub.title)
    except Exception as e:
        continue
votes = np.array([upvotes, downvotes]).T



#adding a number to the end of the %run call will get the ith top post.
%run top_showerthoughts_submissions.py 2
print("Post contents: \n")
print(top_post)

"""
contents: an array of the text from the last 100 top submissions to a subreddit
votes: a 2d numpy array of upvotes, downvotes for each submission.
"""
n_submissions = len(votes)
submissions = np.random.randint( n_submissions, size=4)
print("Some Submissions (out of %d total) \n-----------"%n_submissions)
for i in submissions:
    print('"' + contents[i] + '"')
    print("upvotes/downvotes: ",votes[i,:], "\n")


import pymc3 as pm

def posterior_upvote_ratio( upvotes, downvotes, samples = 20000):
    """
    This function accepts the number of upvotes and downvotes a particular submission recieved,
    and the number of posterior samples to return to the user. Assumes a uniform prior.
    """
    N = upvotes + downvotes
    with pm.Model() as model:
        upvote_ratio = pm.Uniform("upvote_ratio", 0, 1)
        observations = pm.Binomial( "obs",  N, upvote_ratio, observed=upvotes)

        trace = pm.sample(samples, step=pm.Metropolis())

    burned_trace = trace[int(samples/4):]
    return burned_trace["upvote_ratio"]

figsize( 11., 8)
posteriors = []
colours = ["#348ABD", "#A60628", "#7A68A6", "#467821", "#CF4457"]
for i in range(len(submissions)):
    j = submissions[i]
    posteriors.append( posterior_upvote_ratio( votes[j, 0], votes[j,1] ) )
    plt.hist( posteriors[i], bins = 10, normed = True, alpha = .9,
            histtype="step",color = colours[i%5], lw = 3,
            label = '(%d up:%d down)\n%s...'%(votes[j, 0], votes[j,1], contents[j][:50]) )
    plt.hist( posteriors[i], bins = 10, normed = True, alpha = .2,
            histtype="stepfilled",color = colours[i], lw = 3, )

plt.legend(loc="upper left")
plt.xlim( 0, 1)
plt.title("Posterior distributions of upvote ratios on different submissions")

N = posteriors[0].shape[0]
lower_limits = []

for i in range(len(submissions)):
    j = submissions[i]
    plt.hist( posteriors[i], bins = 20, normed = True, alpha = .9,
            histtype="step",color = colours[i], lw = 3,
            label = '(%d up:%d down)\n%s...'%(votes[j, 0], votes[j,1], contents[j][:50]) )
    plt.hist( posteriors[i], bins = 20, normed = True, alpha = .2,
            histtype="stepfilled",color = colours[i], lw = 3, )
    v = np.sort( posteriors[i] )[ int(0.05*N) ]
    #plt.vlines( v, 0, 15 , color = "k", alpha = 1, linewidths=3 )
    plt.vlines( v, 0, 10 , color = colours[i], linestyles = "--",  linewidths=3  )
    lower_limits.append(v)
    plt.legend(loc="upper left")

plt.legend(loc="upper left")
plt.title("Posterior distributions of upvote ratios on different submissions");
order = np.argsort( -np.array( lower_limits ) )
print(order, lower_limits)

def intervals(u,d):
    a = 1. + u
    b = 1. + d
    mu = a/(a+b)
    std_err = 1.65*np.sqrt( (a*b)/( (a+b)**2*(a+b+1.) ) )
    return ( mu, std_err )

print("Approximate lower bounds:")
posterior_mean, std_err  = intervals(votes[:,0],votes[:,1])
lb = posterior_mean - std_err
print(lb)
print("\n")
print("Top 40 Sorted according to approximate lower bounds:")
print("\n")
order = np.argsort( -lb )
ordered_contents = []
for i in order[:40]:
    ordered_contents.append( contents[i] )
    print(votes[i,0], votes[i,1], contents[i])
    print("-------------")


r_order = order[::-1][-40:]
plt.errorbar( posterior_mean[r_order], np.arange( len(r_order) ),
               xerr=std_err[r_order], capsize=0, fmt="o",
                color = "#7A68A6")
plt.xlim( 0.3, 1)
plt.yticks( np.arange( len(r_order)-1,-1,-1 ), map( lambda x: x[:30].replace("\n",""), ordered_contents) )
