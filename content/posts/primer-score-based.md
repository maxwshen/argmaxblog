---
  title: Primer on Score-based Generative Models
  math: mathjax
  summary: An introduction
  date: 2022-08-14
---

The concept of score matching dates back to non-normalized probabilistic modeling, but has recently attracted interests because it offers an elegant paradigm for generative modeling, and has intimate connection with diffusion models. An introduction to the subject can be found [in the second textbook of Kevin Murphy](https://github.com/probml/pml2-book/releases/latest). Here, we describe some of the main modern ideas i	n a simplified fashion, which mostly builds on the works of [Yang Song](https://yang-song.net/).

## A simple score-based generative model

Let's start by establishing [the result](https://arxiv.org/abs/1907.05600). Say that $s\_\theta(x)$ is a learnable function with parameters $\theta$, such as a neural network, operating on a data sample $x$. If we learn $s$ by minimizing the loss function
$$
BSM =\sum\_{x \in \text{data}} \frac{1}{2}||s\_\theta(x)||^2 + \text{tr} \left(\nabla\_x s\_\theta(x) \right),
$$
then the following rule defines a generative model for our data distribution:
$$
x^{(k+1)} = x^{(k)} + \epsilon\; s\_\theta(x^{(k)}) + \sqrt{\epsilon} \, w^{(k)},
$$
where $k$ denotes the time step, $\epsilon$ is a small parameter of our choosing and $w^{(k)}$ is a normally distributed random number. Therefore, we can start from an arbitrary data point  $x\_0$ , then recursively apply the above rule to generate $x^{(0)},\ldots,x^{(k)}$ which (perhaps after a few initial transient steps) would yield data  as they were sampled from our data distribution.

By looking at the above formulae, it is not immediately obvious that the two formulae above would define a valid generative process, although [this indeed works](https://github.com/probml/pyprobml/blob/master/notebooks/book2/24/score\_matching\_swiss\_roll.ipynb). To unravel the mystery and develop intuition, read below.

### Understanding the Basic Score Matching objective

First, we make sense of the $BSM$ objective, which stands for Basic Score Matching objective. Say that $p(x)$ represents our data distribution and $p\_\theta(x)$ is a model we want to fit to the data. The score of the model is defined as
$$
s\_\theta(x) = \nabla\_x \log p\_\theta(x).
$$
Typically, the score appears in the context of energy-based model namely where $p\_\theta(x)=Z\_\theta^{-1} \exp\left( -E\_\theta(x) \right)$; the quantity $E\_\theta$ is called energy and $Z\_\theta$ is the partition function. In this case, the score boils down to the simple relationship $s\_\theta(x) = -\nabla\_x E\_\theta(x).$ 

To learn the score, a loss function that makes intuitive sense is the expectation of the $L^2$ norm of the score difference,
$$
\mathcal J = \frac{1}{2} \int\_{\text{data}} dx\, p(x)\, ||\nabla\_x \log p(x) - \nabla\_x \log p\_\theta(x)||^2 = \frac{1}{2} \int\_{\text{data}} dx\, p(x)\, ||s(x) - s\_\theta(x)||^2,
$$
although this loss function is not very practical, because the score over the data distribution is not an object we have access to! Without additional assumptions, we only have access to samples from the data distribution $p(x)$. We can think of $p(x)$ as an *implicit distribution*: we cannot evaluate the probability density or mass function, we can only draw samples from it. (And in practice, we might not even be able to do that either.)

However, we have:
$$
||s(x) - s\_\theta(x)||^2 = ||s(x)||^2+||s\_\theta(x)||^2 - 2 \,s\_\theta(x)^Ts(x).
$$
The first term can be ignored because it doesn't contain $\theta$  so it's unaffected by the optimization process. Note that the second term already appears in the loss in the previous section. Following [Hyvarinen](https://www.jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf), the third term, under integral, can be rewritten as::
$$
\int\_\text{data} dx\, p(x) 2 \sum\_i s\_\theta(x)\_i\, s(x)\_i = 2 \int\_\text{data} dx\, p(x) \frac{1}{p(x)} \sum\_i s\_\theta(x)\_i\, \frac{\partial p(x) }{\partial x\_i} = 2 \int\_\text{data} dx\ \sum\_i s\_\theta(x)\_i\, \frac{\partial p(x) }{\partial x\_i}.
$$
We can now use integration by parts to move the derivative sign onto the score $s\_\theta(x)$.  Neglecting the boundary term, as customarily done, yields
$$
2 \int\_\text{data} dx\ \sum\_i s\_\theta(x)\_i\, \frac{\partial p(x) }{\partial x\_i} \approx 2 \int\_\text{data} dx\ \sum\_i \frac{\partial s\_\theta(x)\_i }{\partial x\_i} \, p(x) =  \langle \text{tr} \,\nabla\_x s\_\theta(x)\rangle\_\text{data}.
$$
By replacing the expectation with a sample average, we now establish that the original loss $BSM$ is in fact the loss $\mathcal J$. This fact has unraveled (one part of) the mystery: the loss $BSM$, straightforwardily learnable unlike $\mathcal J,$  suggests that $s\_\theta(x)$ signifies the score of an underlying model $p\_\theta(x)$.

### Intuition

We have derived the basic score-matching objective:
$$
BSM =\sum\_{x \in \text{data}} \frac{1}{2}||s\_\theta(x)||^2 + \text{tr} \left(\nabla\_x s\_\theta(x) \right).
$$
To develop intuition, consider when $x$ is one-dimensional: the score function $s(x) = \nabla\_x \log p(x)$ is just the derivative $d \log p(x) / dx$, and we can rewrite the objective as:
$$
BSM =\sum\_{x \in \text{data}}
\frac{1}{2} || \frac{d \log p(x)}{dx} ||^2
+
\frac{d^2 \log p(x)}{dx^2}.
$$
The first term says: make the first derivative zero. The second term says: make the second derivative as negative as possible. These two conditions are satisfied at local maxima. Taken together, these terms encourage the learned $p\_\theta(x)$ to have a local maxima at every observed $x$ in our dataset.

This view has several implications for machine learning. While it's true that basic score matching, when satisfied *everywhere*, will recover the true data distribution, in practice we always learn with finite samples. The intuition lets us reason that:

- We can't really trust a learned score matching model on $x$ that is far from the training set
- Even within the training set, a learned score matching model can fail to recover the correct probability ratio between two $x$ if observed samples are too sparse around these $x$. This is because the model just learns to put a local maxima at each $x$, but the relative probability mass is underspecified. 

### Understanding Langevin MCMC

The BSM loss allows us to learn a model via the score, namely, bypassing learning $p\_\theta(x)$ directly. But, in order to fully define a generative model, we also need a sampling procedure that only relies on $s\_\theta(x)$. The rule in the first section already shows that, but why does it work? The foundational result is that the stochastic differential equation (SDE; sometimes called Langevin's equation or diffusion process),

$$
	dx = f(x) dt + g(x) dw,
$$
admits the following stationary distribution, where $Z$ is a normalization constant (this result can be found in any [stochastic process textbook](https://search.iczhiku.com/paper/ZxayzrjotAdWr9NH.pdf)) 
$$
	p\_s(x) = Z^{-1} \exp\left(\int dx\, f(x)\right).
$$
That means that if we generate data using the SDE we will sample, after a certain transient, from distribution $p\_s(x)$.  Let's now say we want to sample from a learned model $p\_\theta(x)$, what is the right $f$ to use in the SDE? The magical answer is: the score, $f(x)=\nabla\_x \log p\_\theta(x) = s\_\theta(x)$ . In fact, plugging the score into the expression for $p\_s(x)$ yields
$$
	p\_s(x) \propto \exp\left(\int dx\, \nabla\_x \log p\_\theta(x) \right) \propto \exp\left(\log p\_\theta(x) \right) \propto p\_\theta(x).
$$
Hence, to arrive at sampling rule in the first section, all we need is to discretize the SDE by assuming $dt\approx \epsilon$ . The factor $\sqrt \epsilon $ arises because of the scaling law for the Brownian motion. 

## The Denoising Score Matching (DMS) objective

Score matching doesn't work with discrete data, as when the data are not continuously distributed the score is not defined.  For example consider an RGB image where pixel values $x$ take integer values from 0 to 256. Therefore, we have that $p(x)=0$ on every non-integer value, hence $\log p(x) = \infty$ and the score is not defined. To overcome this problem, we add noise to our original dataset $p(x)$ to build new dataset $q(\tilde x)$, where $\tilde x$ indicates a corrupted version of $x$. Specifically:
$$
q(\tilde x) = \int dx\, p(x) q(\tilde x|x).
$$
The transition kernel $q(\tilde x | x)$ denotes a corruption process such as adding Gaussian noise to the image, so that $q(\tilde x = x + \epsilon | x)$ where $\epsilon \in \mathcal N(0, \sigma)$. Because noise is a continuous variable,  every $x$ has now non-zero probability, $p(x)$ is smooth, and the score is finite and differentiable. 

What objective to use to work with the noisy data? Ideally, we would like to adapt the basic score matching objective to noisy dataset, that is:
$$
DSM = \frac{1}{2} \int d\tilde x\, q(\tilde  x)\, ||s\_\theta(\tilde x) - \nabla\_{\tilde x} \log q(\tilde  x)||^2.
$$
[A cool result](http://www.iro.umontreal.ca/~vincentp/Publications/smdae\_techreport.pdf) shows that the above loss can be approximated to
$$
DSM \approx \frac{1}{2} \int dx\, d\tilde x\, p(x) q(\tilde x | x)\, ||s\_\theta(\tilde x) - \nabla\_x \log q(\tilde  x | x)||^2 = \mathbb E\_{q(\tilde x, x)} ||s\_\theta(\tilde x) - \nabla\_x \log q(\tilde  x | x)||^2.
$$
The advantage of the above loss is that the score $\nabla\_x \log q(\tilde x | x)$ is tractable, as we choose its analytical form, unlike $\nabla\_{\tilde x} \log q(\tilde x).$ In fact, if $q(\tilde x | x)$ is Gaussian with variance $\sigma^2$, one can easily arrive at 
$$
DSM\_\sigma \approx \mathbb E\_{q(\tilde x, x)} ||s\_\theta(\tilde x) + \frac{\tilde x - x}{\sigma^2}||^2.
$$
From $DSM\_\sigma$, we see that $s\_\theta(\tilde x) \approx \epsilon/\sigma^2$ where $\epsilon = \tilde x - x$. Therefore, the score returns the rescaled noise added to an image, given as input the corrupted version $\tilde x$. Also $DSM$ has been introduced with the goal of making discrete data continuous, it has also the noteworthy advantage that, as opposed to $BSM$, avoids the trace term in the objective that is computationally expensive. However, by adding noise, we are corrupting our dataset which may yield to trouble whenever $q(\tilde x) \not\approx p(x)$. 

### Adding noise at multiple scales

How to choose the variance $\sigma^2$ of our Gaussian noise, or more generally, the noise strength? If $\sigma$ is too small, then the noise doesn't really change much and we still deal with very low probability regions. If $\sigma$ is to high, then the noise destroys our dataset. 

Considering Gaussian perturbations, a popular approach is to choose $\sigma\_1,\ldots,\sigma\_L$ noise scales, such that the first one is small, $q\_{\sigma\_1}(\tilde x) \approx p(x)$, and the last one is large, $q\_{\sigma\_L}(\tilde x) \approx \mathcal N(x, \sigma\_L)$. The new objective becomes:
$$
\mathcal L = \frac{1}{L}\sum\_{i=1}^L \lambda(\sigma\_i)DSM\_{\sigma\_i}(\theta)
$$
where $\lambda(\sigma)$ weighs differently each $DISM\_\sigma$. [Yang Song and Stefano Ermon, empirically found](https://arxiv.org/abs/2006.09011) that adding multiple noise scales was key to score-generative based modelings [BLOG], and set $\lambda(\sigma) =\sigma$.

## Conclusions

In the last 2 years, the field of score-based generative models has evolved and established a connection with diffusion models. At present, score-based generative models offer diverse generative power comparable to GANs but with much simpler training. However, effcient sampling methods are still lacking. We recommend [this comprehensive and very well made tutorial](https://cvpr2022-tutorial-diffusion-models.github.io/) as a further reading.













