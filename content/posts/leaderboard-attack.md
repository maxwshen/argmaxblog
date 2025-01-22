---
  title: "Outranking leaderboard models on test sets, with only their predictions"
  math: mathjax
  summary: A brainteaser
  date: 2025-01-21
---
<!-- # Prediction contest attacks with access to high-ranked $y$ -->
<!-- # Outranking leaderboard models on test sets, with only their predictions -->

Here's a brainteaser. Suppose there's a Kaggle-style machine learning prediction contest, which works like this:

- Contestants get a training dataset (X and y), and test X
- They submit their predictions of y $\in \mathbb{R}^n$ on the test set to Kaggle
- Kaggle automatically scores these predictions against the hidden test y $\in \mathbb{R}^n$ using a loss function (suppose it's mean-squared error for now; later we discuss cross entropy), and ranks submissions by loss in real-time on a leaderboard

Suppose you've (somehow) obtained the test-set predictions `y_pred` for the #1 ranking model. Furthermore, we'll assume that we have zero information on how `y_pred` is wrong, other than its non-zero MSE. How can you essentially guarantee (i.e., with very high probability) that you'll take the #1 leaderboard spot, without performing any training?


## Solution

Our setup is we have the top-ranked model predictions $y \in \mathbb{R}^n$, the true hidden labels are $y^\dagger \in \mathbb{R}^n$, and the loss function is $\ell(y) = \| y - y^\dagger \|\_2^2$. To take the #1 leaderboard spot, we'll need a submission $z \in \mathbb{R}^n$ satisfying $\ell(z) < \ell(y)$.

The fact that we know nothing about how $y$ is wrong, but that it has non-zero MSE, tells us that $y \neq y^\dagger$, and $y^\dagger$ could be in any direction from $y$ according to a uniform distribution over angles in our $n$-dimensional space.

The strategy is to sample a random unit vector $\epsilon \in \mathbb{R}^n$, and for some small scalar magnitude $c <<1$, construct two prediction vectors for submission: $y + c\epsilon$ and $y - c\epsilon$. With high probability, one of these will take the #1 leaderboard spot. What's great is this solution only requires two submissions, and doesn't require any training or even any actual details of the prediction contest.

This works because $y$ and $\epsilon$ define a hyperplane which divides the $n$-dimensional space into two halfspaces. Most vectors in the space are closer to $y + c\epsilon$ or $y - c\epsilon$ than $y$, except for those in a thin band sandwiched between $y - c\epsilon$ and $y + c\epsilon$. As we shrink the magnitude $c$, this band gets smaller and smaller, increasing the probability that any random vector is closer to either $y - c\epsilon, y + c\epsilon$ than $y$.


## Analysis

Let's analyze the probability of failure. Our setup:

- $n$: The number of datapoints in the test set.
- $y \in \mathbb{R}^n$: The top-ranked model's predictions on the test set.
- $y^\dagger \in \mathbb{R}^n$: The hidden labels on the test set. 
- $m$: the MSE of the top-ranked model predictions. Equal to $\frac{1}{n} \| y - y^\dagger \|\_2^2$.
- $\epsilon$: Our random unit vector
- $c$ The magnitude of our random perturbation $c\epsilon$.

Without loss of generality, we'll translate $y, y^\dagger$ so that $y$ is at the origin. Our strategy fails when $\|y^\dagger - y\|\_2^2 < \| y^\dagger - y \pm c\epsilon \|\_2^2$. Rewriting with $y$ at the origin, the faillure condition is: 

$$
\|y^\dagger\|\_2^2 < \|y^\dagger \pm c\epsilon\|\_2^2.
$$

Where is $y^\dagger$? Its MSE to $y$ (the origin) is $m$, and it is uniformly distributed on angles. Using the MSE $m$, we have that $\|y^\dagger - y\|\_2^2 = \|y^\dagger\|\_2^2 = mn$. Thus, we can express $y^\dagger$ as $\sqrt{mn} \cdot v$, where $v$ is a random unit vector.

Our perturbation $\epsilon$ is also a random vector. We'll expand one of the failure conditions (the - one) as:

$$
\| \sqrt{mn} \cdot v \|\_2^2 < \| \sqrt{mn} \cdot v - c\epsilon\|\_2^2
$$

$$
mn < mn + c^2 - 2c\sqrt{mn} \langle v, \epsilon \rangle
$$

$$
0 < c^2 - 2c\sqrt{mn} \langle v, \epsilon \rangle
$$

$$
2\sqrt{mn} \langle v, \epsilon \rangle < c
$$

The other failure condition (+) ends up being:

$$
-2\sqrt{mn} \langle v, \epsilon \rangle < c
$$

Combining the two, our failure condition is:

$$
|\langle v, \epsilon \rangle| < \frac{c}{2\sqrt{mn}}
$$

Our $v, \epsilon$ are random unit vectors. It turns out that the random variable $\langle v, \epsilon \rangle$'s distribution over [-1, 1] is linearly related to Beta((N-1)/2, (N-1)/2) distribution over [0, 1]:[^ref]

[^ref]: https://stats.stackexchange.com/questions/85916/distribution-of-scalar-products-of-two-random-unit-vectors-in-d-dimensions

$$
\langle v, \epsilon \rangle = 2y-1
$$

where

$$
y \sim \textrm{Beta}\left( \frac{n-1}{2}, \frac{n-1}{2} \right).
$$

With finite $n$, we can use code to calculate the probability of failure using the Beta cdf, as a function of $m,n,c$:

```python
from scipy.stats import beta

def prob_failure(n: int, c: int, mse: float):
    """ Compute probability of failure (y being closest,
        and not y plus/minus c\epsilon) given:
        n: dimension
        c: perturbation magnitude
        mse: MSE between y and y*
    """    
    # Transform bounds from [-c/(2√m), c/(2√m)] to [0,1]
    # x -> (x+1)/2 transforms [-1,1] to [0,1]
    # So our bounds transform from ±c/(2√m) to 0.5 ± c/(2√m)
    bound = c/(2*np.sqrt(n*mse))
    lower = 0.5 - bound
    upper = 0.5 + bound
    
    return beta.cdf(upper, (n-1)/2, (n-1)/2) - \
           beta.cdf(lower, (n-1)/2, (n-1)/2)
```

Here are some examples, all at MSE = 0.1. Entries are the probability of *success*, for MSE=0.1, and varying $n$, the test set size, and $c$, the perturbation magnitude.

|    n |   c=0.01 |   c=0.03 |   c=0.10 |   c=0.30 |
|-----:|---------:|---------:|---------:|---------:|
|   10 | 0.976721 | 0.930227 | 0.769875 | 0.370083 |
|  100 | 0.974963 | 0.924985 | 0.753576 | 0.345334 |
| 1000 | 0.974792 | 0.924476 | 0.752004 | 0.343035 |

Interestingly, the probability of success doesn't really depend on $n$ at all! The optimal $c$ seems to depend primarily on the MSE. 

We can reason about this because the distribution of $\langle v, \epsilon \rangle$ also approaches a Gaussian $\mathcal{N}(0, 1/n)$ as $n \rightarrow \infty$, which gives a nice intuitive picture: as $n$ increases, the Gaussian becomes narrower with standard deviation $\sqrt{n}$, but the failure "band" width $\frac{c}{2\sqrt{mn}}$ also shrinks with $\sqrt{n}$: these effectively cancel out. This is great because often, increasing dimensionality causes problems. This strategy's success doesn't depend on dimensionality at all.

This is great and all, but MSE is only one loss function used in machine learning. Could this strategy also work for other loss functions?

## Cross entropy

Unlike MSE, cross entropy is not a proper distance metric, and we no longer have a geometric intuition of halfspaces to see why the strategy might still work.

As such, it's a surprise that it does still seem to work. I'll show this empirically by simulation, and using a rough analysis with a first-order Taylor expansion.

In this setup, we consider random probability vectors $y, y^\dagger$, and cross entropy:

$$
H(y^\dagger, y) = -\sum\_i y\_i^* \log(y)
$$

We compare three probability vectors: $y$, $y + c\epsilon$, and $y - c\epsilon$, where we encourage the latter two to also be probability vectors by defining $\epsilon = v - y$ for a probability vector $v$ uniformly distributed in the simplex. For simplicity's sake, we'll ignore boundary conditions and just assume that $y + c\epsilon$ and $y - c\epsilon$ are valid probability vectors.

We seek to compare:

- $H(y^\dagger, y)$
- $H(y^\dagger, y + c\epsilon)$
- $H(y^\dagger, y - c\epsilon)$

Plugging in, we have:

$$
H(y^\dagger, y + c\epsilon) = -\sum\_i y\_i^* \log (y\_i + c\epsilon\_i)
$$

The first-order Taylor expansion of the cross-entropy of the perturbed predictions is:

$$
H(y^\dagger, y + c\epsilon) \approx -\sum\_i y\_i^* ( \log(y\_i) + c\epsilon\_i / y\_i )
$$

So the difference

$$
H(y^\dagger, y) - H(y^\dagger, y + c\epsilon) \approx c\sum\_i \frac{y\_i^* \epsilon\_i}{y\_i}
$$

Similarly,

$$
H(y^\dagger, y) - H(y^\dagger, y - c\epsilon) \approx -c\sum\_i \frac{y\_i^* \epsilon\_i}{y\_i}
$$

Thus, up to the first-order Taylor expansion which becomes more realistic as $c \rightarrow 0$, *one* of $y \pm c\epsilon$ improves cross entropy.

### Simulation

```python
import numpy as np
from collections import defaultdict

d = 10

def sample_p(d):
    sample = np.random.uniform(size = d)
    return sample / sample.sum()


def dist(x, y):
    """ cross entropy """
    return sum(yi * np.log(xi) for xi, yi in zip(x, y))


for eps_factor in [10, 100, 1000]:
    results = defaultdict(int)
    for n in range(1000):

        x = sample_p(d)
        y = sample_p(d)
        eps = sample_p(d)

        perturb_minus = lambda x, eps: (x - eps / eps_factor) / (x - eps / eps_factor).sum()
        perturb_plus = lambda x, eps: (x + eps / eps_factor) / (x + eps / eps_factor).sum()

        dists = [
            dist(x, y),
            dist(perturb_minus(x, eps), y),
            dist(perturb_plus(x, eps), y)
        ]

        if dists.index(min(dists)) != 0:
            results[True] += 1
        else:
            results[False] += 1

    print(eps_factor)
    print(results)
```

Using this code, I get the results:

|    eps_factor |   % success |
|-----:|---------:|
|   10 | 64.4% |
|  100 | 94.6% |
| 1000 | 99.6% |

Thus empirically, the same strategy seems to hold for cross entropy.

<!-- What's intriguing is this also holds for cross-entropy, even though cross-entropy isn't a proper distance metric, unlike mean-squared error which is the $L_2$-norm. -->