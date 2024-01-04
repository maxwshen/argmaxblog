---
 title: "Treating high-dimensional prediction/generation as optimization"
 math: mathjax
 description: Exploring connections between diffusion models, alphafold's recycling, structured prediction energy networks, and manifold learning. 
 summary: Exploring connections between diffusion models, alphafold's recycling, structured prediction energy networks, and manifold learning. 
 date: 2022-07-01
---

Are diffusion models a passing fad, or are they here to stay? This post arose, in part, from pondering this question ...

One perspective on the history of machine learning is a story of increasing ability to solve problems with high-dimensional inputs $x$, and models with high-dimensional parameters $\theta$. Consider the progression from Yann LeCun's LeNet in 1989 on MNIST (28x28 pixels) to 2012's AlexNet on ImageNet (256x256 pixels), enabled by scaling convolutional nets trained with stochastic gradient descent using GPUs. But in textbook problems like regression or classification, the output space $\mathcal{Y}$ is typically low dimensional or simple.

Since 2012, deep learning research has had increasing interest in handling high-dimensional outputs $y$. There are many ways to create high-dimensional outputs, such as using transposed convolutions, autoregressive sampling, and normalizing flows. But two unrelated yet highly impactful advances: diffusion modeling (used in DALL-E) and AlphaFold, share in common another approach that I find particularly compelling: viewing high-dimensional prediction or generative modeling as optimization performed iteratively.

In this post, I provide a unifying view on diffusion models, AlphaFold2's recycling mechanism (found to be fairly important in ablation studies), and human pose estimation by iterative error feedback in terms of iterative optimization. I relate them to structured prediction energy networks and score-based modeling. Finally, I describe pros and cons of iterative optimization over single-pass modeling for high-dimensional output in terms of manifold learning, ease of constructing large supervised training data, and robustness to prediction errors.

#### Outputting in high-dimensions with iterative optimization

In our setting, we consider machine learning procedures that map from input space $\mathcal{X}$ to output space $\mathcal{Y}$ which we presume to have high dimension. In diffusion image modeling, $x$ is a pixel-wise Gaussian distribution and $y$ is a high-resolution color image. In AlphaFold2, $x$ is a multiple DNA sequence alignment and $y$ is a 3D protein structure. For convenience, we consider $\mathcal{Y}$ as $\mathbb{R}^d$ from here on. 

**Single-pass:** We learn a function $f\_\theta: \mathcal{X} \rightarrow \mathcal{Y}$. To run a forward pass on some $x$, we return $f(x)$. Given a training dataset $(X, Y)$, we learn $f\_\theta$ as $\arg\min\_{f} \ell(f(x), y)$ for some loss $\ell$.

**Iterative optimization:** We learn a function $f\_\theta: \mathcal{X}, \mathcal{Y} \rightarrow \mathcal{Y}$, where we view $f\_{\theta}(x,y) = e$ where $e \in \mathcal{Y}$ as an "edit", and $f\_{\theta}(x,y)$ can be understood as a vector field on $\mathcal{Y}$. To run a forward pass on some $x$, we initialize a $y\_0$, iteratively update
$$
\tag{1}
y\_{t+1} \leftarrow y\_t + f\_\theta(x, y\_t)
$$

for $T$ steps, then return $y\_T$. This has an enticing similarity to gradient ascent that we will comment on later. 

#### Learning to edit with constructed training data

Let's use the setting of the seminal paper [Human Pose Estimation with Iterative Error Feedback](https://arxiv.org/abs/1507.06550) (Carreira et al., 2016) to motivate why we might want such models, and see how we can train them.

The task is: given $x$ a 2D image of a human, predict $y$ which is a set of 2D coordinates representing the human's pose (e.g., positions of the torso, hands, feet). A key motivation for iterative optimization is that human poses have structure: torsos are physically connected to hands, so their relative positions are not arbitrary. A model $f\_\theta: \mathcal{X} \rightarrow \mathcal{Y}$ may not output $y$ that respects this structure, but a model that receives both $x, y$ as input may be able to easily correct mistakes. 

With only access to a ground truth image $x$ and pose $y$, training data of good edits can be obtained by sampling unseen poses $\hat{y}$, and computing edits $e = y - \hat{y}$. Training then aims for $\arg\min\_f \ell(f(x, \hat{y}), e)$. This paper bounds the norm of each edit, treating them as directions, and finds better performance from taking multiple small steps rather than directly jumping to the output in a single step. 

```python
def sample\_training\_edit(x, y):
    yhat = sample\_y()
    edit = y - yhat   # yhat + edit = y
    return (x, yhat), edit
```

What should we initialize $y\_0$ to? Given our constructed training procedure, this is a critical question when $\mathcal{Y}$ is high-dimensional: there are exponentially many directions around any $y$ where we can sample edits. Fortunately, we can control $y\_0$ during training and testing. Carreira choose $y\_0$ as the median coordinate of each limb in the training set. As we shall see later, diffusion models sample $y\_0$ from a unit Gaussian, and AlphaFold2 uses a "black hole" initialization of starting every atom at the origin.

#### Iterative optimization as gradient ascent on an induced energy landscape 

If we presume a probability distribution $p(y|x)$ over $\mathcal{Y}$, our edits can sometimes be idealized as the *score function* $e=f\_\theta(x, y) = \nabla\_y \log p(y|x)$, and our editing procedure can be viewed as gradient ascent: $y\_{t+1} \leftarrow y\_t + \nabla\_y \log p(y|x)$. 

This is equivalent to optimizing over an energy landscape, where non-negative energy $E(y|x)$ is unnormalized probability: $p(y|x) = E(y|x)/Z(x)$ with $Z(x) = \int\_{\mathcal{Y}} E(y|x) dy$. This equivalence arises because the [score function](https://www.jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) does not depend on the partition function: $\nabla\_y \log p(y|x) = \nabla\_y \log E(y|x)$. 

Proof:
$$
\begin{aligned}
\nabla\_y \log p(y|x) &= \nabla\_y (\log E(y|x) - \log Z(x)) \\\
&= \nabla\_y \log E(y|x) - \nabla\_y \log Z(x) \\\
&= \nabla\_y \log E(y|x).
\end{aligned}
$$
It turns out this style of iterative optimization can sometimes be interpreted as gradient ascent on some energy landscape. At least these two conditions must be satisfied.

First, we can employ a theorem of local consistency ([Hyvarinen, 2005](https://www.jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)). Suppose two distributions $p\_1, p\_2$ over a space $\mathcal{Z}$ where $p\_1(z) >0$ and $p\_2(z) > 0$ for all $z \in \mathcal{Z}$. Then, $\nabla\_z \log p\_1(z) = \nabla\_z \log p\_2(z)$ for all $z \in \mathcal{Z}$ if and only if $p\_1(z) = p\_2(z)$ for all $z \in \mathcal{Z}$. This means that a score function *induces* a probability distribution (equivalently, an energy landscape).

Proof sketch ($\implies$): Suppose $\nabla\_z \log p\_1(z) = \nabla\_z \log p\_2(z)$ for all $z$. Then, integrating, we have $\log p\_1(z) = \log p\_2(z) + c$ for all $z$. However, each pdf must sum to 1 over the whole space, so $c$ must be 0; qed. Proof for $\impliedby$ is trivial.

Second, the score function, or gradient field, must be a conservative vector field for it to be a valid gradient of a scalar-valued function $p(y|x)$. When the score is the prediction of an unbounded neural network, this is generally not the case.

**[Structured prediction energy networks](https://arxiv.org/pdf/1511.06350.pdf) (SPENs).** SPENs can be understood as an alternative parametrization of our iterative optimization framework. A SPEN learns $E\_\theta: \mathcal{X}, \mathcal{Y} \rightarrow \mathbb{R}\_{>0}$ as a deep neural network where the output is non-negative energy. To iteratively predict $y$, the SPEN uses backpropagation to compute the gradient $\nabla\_y E\_\theta (x,y)$ at each step to update $y$ to optimize its predicted energy $E\_\theta(x, y)$. This is more expensive than our parametrization, where a single forward pass of $f\_\theta$ directly provides the gradient. [SPENs](https://arxiv.org/pdf/1703.05667.pdf) can be learned with [implicit differentiation](https://argmax.ghost.io/implicit-differentiation-through-equilibria/): we learn an energy landscape such that the local minima obtained from our iterative updates matches target $y$, using the gradient of the steady state w.r.t. energy landscape parameters $\theta$.

**[AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2)'s Recycling.** Iterative refinement is a key concept in AlphaFold that is used at several scales: within the structure module, and also on the whole network where it's termed recycling. In recycling, rather than predicting edits that are added to the current $y$, the outputs $y$ (such as predicted backbone atom coordinates) are directly fed back into the learned network. Recycling is trained to minimize the average loss of each $y$ obtained over $N$ cycling iterations, and an unbiased Monte Carlo approximation is used to estimate the loss that is more efficient than always performing $N$ steps: a number of iterations $1 \leq N' \leq N$ is sampled, and gradient flow is stopped for all but the last iteration. In ablation studies (Fig. 4), recycling is found to "contribute markedly to accuracy". 

**Diffusion models.** In our framework of iterative optimization, a diffusion model (in unconditional generative modeling) is a process for transforming noise to output $y$ as:
$$
\texttt{(noise) } y\_0 \rightarrow y\_{1} \rightarrow ... \rightarrow y\_{T-1} \rightarrow y\_{T} \texttt{ (data)}
$$
where all $y\_0, ..., y\_T$ are in the same space; for simplicity let's take this as $\mathbb{R}^d$. The initial $y\_0$ is conventionally a unit Gaussian distribution. To construct training data from an observed $y\_T$, the series $y\_{T-1}, ... y\_0$ is obtained by adding increasing amounts of Gaussian noise to $y\_T$. By properties of Gaussian noise, this is equivalent to obtaining $y\_{t-1}$ by adding scaled unit Gaussian noise to $y\_t$ for each $t$: $y\_{t-1} = y\_t + \epsilon\_t$ where $\epsilon\_t \propto \mathcal{N}(0, I)$.

Diffusion model training is typically motivated as optimizing a variational bound on the negative log likelihood, but in practice it amounts to predicting $y\_t$ given $y\_{t-1}$. A favored parametrization popularized by [denoising diffusion probabilistic models](https://arxiv.org/pdf/2006.11239.pdf) is to predict the edit $\hat{\epsilon}\_t$ given $y\_t$ instead. Data generation is similar to algorithm 1, but with additional Gaussian noise. Starting at $y\_0$ sampled from a Gaussian, we iterate:
$$
y\_{t+1} = y\_t - c\_1 \hat{\epsilon}\_t(y\_t) + c\_2z\\\z \sim \mathcal{N}(0, I)
$$
where $c\_1, c\_2$ depend on the user-specified noise schedule used in constructing training data. This can be understood as Langevin dynamics where the predicted edit $\hat{\epsilon}$ is the gradient of the data density.

#### Iterative optimization *vs* single-pass models

The main disadvantage of iterative optimization is slower speed for producing outputs. In some areas, this is a significant drawback. But here are some advantages:

- **Manifold learning:** When $y$ is high-dimensional yet structured, the manifold hypothesis says that real $y$ lives on a relatively lower-dimensional manifold. By constructing edits from observed samples on the manifold, we can easily learn to edit towards the manifold since arbitrary $y$ are most likely not on the manifold. In a single-pass model $f\_\theta: \mathcal{X} \rightarrow \mathcal{Y}$, the model might erroneously predict some $y$ off the manifold, when there could be easily learned edits to put $y$ closer to the manifold. Models that see $y$ as input might have an easier time creating outputs that stay on or near the manifold.
- **Using more information:** A learned function $e = f\_\theta(x, y)$ can produce a useful output using information in $x$ or in $y$, in contrast to a single-pass model that only depends on $x$.
- **Ease of constructing large, supervised training data:** If there's anything that 2012-now has taught us that deep learning is good at, it's handling large supervised training data. This strength of deep learning seems well aligned with the ease of iterative optimization approaches to generate lots of supervised data. Consider [predicting 3D RNA structures](https://www.science.org/doi/10.1126/science.abe5650): due to their instability compared to proteins, only 18 three-dimensional structures have been measured, but by sampling structures and aiming to predict distance to known structures, supervised training on 10,000s of data points was achieved. (In predicting distance rather than edits, this can be viewed as a structured prediction energy network.)

- **Easier learning task, and robustness to prediction errors.** Consider a convex energy landscape $q(y)$: to find the global minima, a single-pass model has only one shot, and it must be rather precise to land anywhere near the minima when $\mathcal{Y}$ is high-dimensional. In contrast, our predicted edit $f\_\theta(y)$ is a valid descent direction as long as $\langle f\_\theta(y), \nabla\_{y} q(y) \rangle < 0$ where $\langle, \rangle$ is the inner product: there exists a step size $\lambda$ where updating $y \rightarrow y - \lambda f\_\theta(y)$ reduces $q(y)$. If we take enough steps, then we will reach the global minima. Notably, at every point, 50% of all possible directions are valid descent directions! Our learned edits can be rather imprecise, and our final output is more robust to prediction errors.

**The power of approximate gradients**. A view I personally enjoy is noting the strong resemblance with stochastic gradient descent, which works surprisingly well in practice. From this view, iterative optimization is just taking something we know works well and deep learning is good at - optimizing things with approximate gradients - and uses it for creating high-dimensional outputs.

#### Design choices in iterative optimization

- Choice of $y\_0$. In Carreira's human pose estimation, this was initialized to the median limb coordinate over the training data. AlphaFold2 uses a black-hole initialization of starting every atom at the origin, while diffusion models initialize at a unit Gaussian sample. In general, we know that choosing a specific direction for learning optimization is helpful, otherwise there are exponentially many directions to learn to edit from. And deep neural nets are certainly flexible enough to learn to optimize from the origin. But might there be other, better, strategies for choosing $y\_0$, perhaps in a data-dependent or task-dependent manner? What happens when $y\_0$ is already on the manifold?
- Choice of edits. Carreira and AlphaFold2 aim to learn to directly edit towards observed data, while diffusion models aim to edit towards a noised version of observed training data. Is there merit to exploring other editing strategies that do not point directly towards training data?
