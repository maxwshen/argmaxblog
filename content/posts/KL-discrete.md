---
 title: "On KL Divergence in Discrete Spaces"
 math: mathjax
 author: ["Max Shen", "Nathaniel Diamant"]
 summary: KL behaves differently in discrete spaces than in continuous spaces.
 date: 2022-05-01
---

The Kullback-Leibler Divergence between two distributions $P$ and $Q$ is $D\_{\texttt{KL}}(P \| Q) = \sum\_{x \in \mathcal{X}} P(x) \log(\frac{P(x)}{Q(x)})$ when $P, Q$ are discrete distributions, and $D\_{\texttt{KL}}(P \| Q) = \int\_{-\infty}^{\infty} P(x) \log(\frac{P(x)}{Q(x)}) dx$ when $P, Q$ are continuous distributions. The KL Divergence is asymmetric: $D\_{\texttt{KL}}(P \| Q) \neq D\_{\texttt{KL}}(Q \| P)$ in general. In machine learning, we typically consider $P$ to be the fixed target or reference distribution, and $Q$ to be the learned, approximating distribution that we wish to make "close" to $P$. 

$D\_{\texttt{KL}}(P \| Q)$ is known as the forward KL, and $D\_{\texttt{KL}}(Q \| P)$ is the reverse KL. So, what's the difference between forward and reverse KL?

# Continuous spaces

In continuous spaces, it is well known that forward KL is *mode-covering*, while reverse KL is *mode-seeking*. 

- When learning $Q$ to minimize the forward KL divergence, $D\_{\texttt{KL}}(P \| Q)$, $Q$ aims to be *inclusive* and *cover all the modes* of $P$. It *includes* all modes.
- When learning $Q$ to minimize the reverse KL divergence, $D\_{\texttt{KL}}(Q \| P)$, $Q$ aims to be *exclusive* and *cover only a subset of modes* of $P$. It *excludes* modes.

This property has been visualized and described by many blog posts, for instance [here](https://www.tuananhle.co.uk/notes/reverse-forward-kl.html) and [here](https://agustinus.kristia.de/techblog/2016/12/21/forward-reverse-kl/). 

*Aside:* PyTorch's KLDivLoss function $L(y\_{\texttt{pred}}, y\_{\texttt{true}})$ is the forward KL. 

*Another aside*: The KL divergence in the variational inference ELBO is the [reverse KL](https://mpatacchiola.github.io/blog/2021/01/25/intro-variational-inference.html)! That is, we maximize the lefthand side of 

$$\mathbb{E}\_q [ \log p(x,z) - \log q(z) ] = \log p(x) - D\_{\texttt{KL}}(q(z)\|p(z|x))$$

where the right side is the evidence lower bound, and the right side has the reverse KL between the approximating distribution and the posterior. But also, the objective (left-hand side) can be written as maximizing $\mathbb{E}\_q [ \log p(x|z) ] - D\_{\texttt{KL}}(q(z)\|p(z))$: the reverse KL between the approximating distribution and the prior. 

# Discrete spaces

Can we obtain a better understanding of the forward and reverse KL in discrete spaces? Do the properties of mode-seeking *vs* mode-covering also apply in discrete spaces? Let's explore.

First, we define our setting. We consider a discrete space with $K$ dimensions. When working with discrete spaces, it is conventional to specify a multinomial distribution, where the probability mass on each item is essentially independent (with information sharing only through the constraint that total probability sums to 1). No matter how many "modes" $P$ has, when $Q$ is a multinomial, it can freely fit them all, no matter if the KL is forward or reverse.

This is in contrast to continuous spaces where $Q$ is a simple distribution. For instance, if $Q$ is a 1-D Gaussian distribution, the probability density at point $x=0$ is intimately related to the probability density at $x=0.1$. 

## Optimizing discrete $Q$: forward KL $\propto$ Multinomial Log-Likelihood

Recall that $\textrm{MultinomialPMF}(\mathbf{x}|n, \mathbf{p}) = \frac{n!}{\prod\_j x\_j!} \prod\_i p\_i^{x\_i}$ for $K$ mutually exclusive events where event $i$ occurs with probability $p\_i$ and $\sum\_i {p\_i}=1$, and $\mathbf{x}$ is a vector where $x\_i$ is the integer number of times that event $i$ was observed, with $\sum\_i x\_i = n$ total observations.
$$
\begin{aligned}
\textrm{MultinomialPMF}(\mathbf{x}|n, \mathbf{p}) &= \frac{n!}{\prod\_j x\_j!} \prod\_i p\_i^{x\_i} \\\
\log \textrm{MultinomialPMF}(\mathbf{x}|n, \mathbf{p}) &= \sum\_i x\_i \log p\_i + \log{n!} - \sum\_j{\log x\_j!} \\\
\frac{1}{n}\log \textrm{MultinomialPMF}(\mathbf{x}|n, \mathbf{p}) &= \underbrace{\sum\_i \frac{x\_i}{n} \log p\_i}\_{-\texttt{crossentropy}(\frac{\mathbf{x}}{n} \| \mathbf{p})} + \underbrace{\frac{1}{n}\Big(\log{n!} - \sum\_j{\log x\_j!}\Big)}\_{\textrm{not a function of } \mathbf{p}}
\end{aligned}
$$
Using the decomposition of forward KL Divergence into entropy and cross-entropy, we have:
$$
\begin{aligned}
D\_{\texttt{KL}}(P \| Q) &= \sum\_{i} P\_i \log \frac{P\_i}{Q\_i} \\\
&= \underbrace{\sum\_{i} P\_i \log P\_i}\_{\textrm{-Entropy of } \mathbf{P}}  \underbrace{-\sum\_{i} P\_i \log Q\_i}\_{\texttt{crossentropy}(\mathbf{P} \| \mathbf{Q})} \\\
\end{aligned}
$$
We take $Q$, our learned distribution, as the multinomial probability parameter $\mathbf{p}$, and take $P$, the fixed target distribution as fixed $\mathbf{x}/n$, the observations from the multinomial. To do this safely, this requires assuming that probabilities in $P$ are rational. In general, just knowing $P$ does not uniquely identify $\mathbf{x}, n$, but if we know $P, \mathbf{x}, n$, then we can write:
$$
D\_{\texttt{KL}}(P \| Q) = - \frac{1}{n} \log \textrm{MultinomialPMF}(\frac{\mathbf{x}}{n}= P|n, \mathbf{p}=Q) - \underbrace{ H(P) + \frac{1}{n} \Big( \log n! - \sum\_j \log x\_j! \Big) }\_{\textrm{not a function of Q}}
$$
where $H(P) = -\sum\_i P\_i \log P\_i$ is the entropy of $P$. From the perspective of learning $Q$, we can ignore terms that are not a function of $Q$, so we get:
$$
D\_{\texttt{KL}}(P \| Q) \propto - \log \textrm{MultinomialPMF}(\frac{\mathbf{x}}{n}= P|n, \mathbf{p}=Q) + c.
$$
In particular, this means that as a function of $Q$, forward KL and multinomial NLL are perfectly rank-correlated: for any $Q, Q'$, $D\_{\texttt{KL}}(P \| Q) < D\_{\texttt{KL}}(P \| Q')$ *if and only if* $- \log \textrm{MultinomialPMF}(\frac{\mathbf{x}}{n}= P|n, \mathbf{p}=Q) < - \log \textrm{MultinomialPMF}(\frac{\mathbf{x}}{n}= P|n, \mathbf{p}=Q')$. 

This correspondence is pretty satisfying!

## Forward KL and the Dirichlet distribution

It turns out a similar decomposition can be performed with the Dirichlet distribution instead. 

Recall that the Dirichlet distribution has a concentration parameter $\alpha$, which is a vector with $K$ elements which are positive real numbers. Its support is over the $K$-dimensional simplex, and its probability density function is:
$$
\textrm{DirichletPDF}(\mathbf{x}| \mathbf{\alpha}) = \frac{\Gamma(\sum\_i \alpha\_i)}{\prod\_j \Gamma(\alpha\_i)} \prod\_i x\_i^{\alpha\_i -1}
$$
Taking the log, we have:
$$
\log \textrm{DirichletPDF}(\mathbf{x}| \mathbf{\alpha}) = \sum\_i (\alpha\_i -1) \log x\_i - \log \mathrm{B}(\alpha)
$$
where $\mathrm{B}$ is the [multivariate Beta function](https://en.wikipedia.org/wiki/Beta\_function) and does not depend on $\mathbf{x}$. When $\alpha-1$ is a valid distribution (it sums to 1), the first term is negative $\texttt{crossentropy}(\alpha-1 \| \mathbf{x})$. When we take $Q$, our learned distribution, as Dirichlet observations $\mathbf{x}$ and take $P$, our fixed target distribution as $\alpha-1$, we have:
$$
\begin{aligned}
D\_{\texttt{KL}}(P \| Q) &= -H(P) \underbrace{-\sum\_{i} P\_i \log Q\_i}\_{\texttt{crossentropy}(\mathbf{P} \| \mathbf{Q})} \\\ 
&= -\log \textrm{DirichletPDF}(\mathbf{x}=Q|\alpha = P+\mathbf{1}) - H(P) - \log \mathrm{B}(P+\mathbf{1}) \\\
\end{aligned}
$$
As a function of $Q$, the forward KL can be simplified by grouping terms that do not depend on $Q$ into a constant $c$:
$$
= -\log \textrm{DirichletPDF}(x=Q|\alpha = P+\mathbf{1}) + c
$$
This formulation is a bit nicer than the multinomial:

- there's no $n$, number of observations, to worry about, and the decomposition is directly equivalent rather than proportional.
- While access to just $P,Q$ is not enough information to convert the KL into the multinomial NLL (because we do not know $n$), it *is* enough information to convert the KL into the dirichlet NLL! This is because $c=-H(P)-\log \mathrm{B}(P+1)$ are easy to compute given $P$.

- In the Dirichlet formulation, both $P,Q$ are treated as positive real numbers rather than requiring an assumption of rational numbers. 

The $+\mathbf{1}$ term in $\alpha$ seems intriguing! How can we understand this?

**Mode:** A property of the Dirichlet distribution when $\alpha\_i >1$ is that the distribution is unimodal, with a mode at a vector $\mathbf{x}$ where each entry $x\_i = \frac{\alpha\_i-1}{\sum\_{k=1}^K \alpha\_k -K}$. When $P$ is in the simplex, each entry of $P+\mathbf{1}$ is greater than 1 by construction, so the single mode occurs at $\mathbf{x}=P$. This is consistent with the known fact that the KL divergence is globally minimized when $Q=P$. So from this perspective, the $+\mathbf{1}$ term is exactly necessary to ensure that the the KL matches $Q$ towards $P$. 

The $+\mathbf{1}$ term also impacts the Dirichlet mean, relative to its mode. 

**Mean:** The mean of the Dirichlet distribution is a vector $\mathbf{x}$ where $\mathbb{E}[x\_i] = \frac{\alpha\_i}{\sum\_{k=1}^K \alpha\_k}$. In our Dirichlet where $\alpha=P+\mathbf{1}$, the mean occurs at the vector $P+\mathbf{1}$, which can be understood as $P$ mixed with a uniform distribution: it has higher entropy than $P$. 

The mean is always closer to the uniform distribution than the mode. This difference can be understood as *skew* in the distribution's likelihood. This observation of skew inspires us to suggest a conjecture. For some fixed $P$, consider the level set of distributions $Q$ with identical values of $D\_{\texttt{KL}}(P \|Q)$, or equivalently identical values of $-\log \textrm{DirichletPDF}(x=Q|\alpha = P+\mathbf{1})$. We conjecture that the entropy of the element-wise averaged distribution in the level set is greater than or equal to the entropy of $P$, with equality only when $Q=P$.

## Another Dirichlet decomposition of the forward KL

$$
\begin{aligned}
\log \textrm{DirichletPDF}(\mathbf{x}| \mathbf{\alpha}) &= \sum\_i (\alpha\_i -1) \log x\_i - \log \mathrm{B}(\alpha) \\\
&= \sum\_i \alpha\_i \log x\_i - \sum\_i \log x\_i - \log \mathrm{B}(\alpha)
\end{aligned}
$$

Then,
$$
\begin{aligned}
D\_{\texttt{KL}}(P \| Q) &= -H(P) \underbrace{-\sum\_{i} P\_i \log Q\_i}\_{\texttt{crossentropy}(\mathbf{P} \| \mathbf{Q})} \\\ 
&= -\log \textrm{DirichletPDF}(\mathbf{x}=Q|\alpha = P) - \sum\_i \log Q\_i - H(P) - \log \mathrm{B}(P) \\\
&= -\log \textrm{DirichletPDF}(\mathbf{x}=Q|\alpha = P) - \sum\_i \log Q\_i +c
\end{aligned}
$$
Let $U$ be the uniform distribution in our $K$-dimensional space. We have:
$$
\begin{aligned}
D\_{\texttt{KL}}(U\|Q) &= -H(U) - \frac{1}{K}\sum\_i \log Q\_i \\\
K \Big(H(U) - D\_{\texttt{KL}}(U\|Q) \Big) &= \sum\_i \log Q\_i
\end{aligned}
$$
Using this, we can write the forward KL, as a function of $Q$, as
$$
\begin{aligned}
D\_{\texttt{KL}}(P \| Q) &= -\log \textrm{DirichletPDF}(\mathbf{x}=Q|\alpha = P) - KD\_{\texttt{KL}}(U\|Q) +c
\end{aligned}
$$
This formulation is a bit more complicated: as $\alpha=P$ which is element-wise less than 1, this Dirichlet now operates in the "rich get richer" regime and has $K$ modes or local optima of the NLL, one at each corner of the simplex - alone, this term encourages "mode-seeking" behavior, encouraging $Q$ to capture just one mode of $P$. However, this effect is counterbalanced by the $KD\_{\texttt{KL}}(U\|Q)$ term (recall that $K$ is the number of dimensions in our discrete space). This term is the forward KL encouraging $Q$ to match the uniform distribution $U$, encouraging $Q$ to have higher entropy. 

As a function of $Q$, these two opposing forces must balance perfectly such that the unique global optima of the KL is achieved $Q=P$, a known property of the KL.

## Connecting the multinomial and dirichlet views


In the multinomial and dirichlet formulations, $Q$ can be understood as the same object: the parameters of a multinomial distribution. This is immediately apparent for the forward KL. For the backward KL, data $\mathbf{x}$ from the Dirichlet distribution can also be interpreted as the parameters of a multinomial distribution, since the Dirichlet is the conjugate prior to the multinomial. 
$$
\textbf{Dirichlet }\alpha: P+1 \textrm{ (fixed target)}\\
\downarrow \\\ 
\textbf{Multinomial } p: Q \textrm{ (learned)}\\ 
\uparrow \\\
\textbf{Normalized count data }x : P \textrm{ (fixed target)}
$$

## Reverse KL in discrete spaces?

We were able to draw connections between the forward KL in discrete space with the multinomial and dirichlet distributions. What about the reverse KL? A quick rewriting of the reverse KL obtains:
$$
\begin{aligned}
D\_{\texttt{KL}}(Q \| P) &= \sum\_{i} Q\_i \log \frac{Q\_i}{P\_i} \\\
&= -H(Q) \underbrace{-\sum\_{i} Q\_i \log P\_i}\_{\texttt{crossentropy}(\mathbf{Q} \| \mathbf{P})} \\\
\end{aligned}
$$
As a function of $Q$, minimizing the reverse KL also aims to reduce the entropy of $Q$. This additional optimization term already makes the reverse KL less savory than the forward KL. Further, in our explorations, we weren't able to find a clean formulation of the reverse KL in terms of likelihoods, since it turns out that the normalizing terms in the multinomial and dirichlet distributions would be a function of $Q$. 

A clearer understanding of the properties of the reverse KL in discrete space would be nice, especially given the reverse KL's central role in variational inference, but this particular road seems to be a deadend. 

---

## Mode-seeking behavior in discrete spaces?

The same intuitive argument that reverse KL in continuous spaces has mode-seeking behavior also applies just as well in discrete spaces. The argument is: suppose some element $P\_i >0$, then the $i$-th term of the reverse KL $Q\_i \log \frac{Q\_i}{P\_i}$ can be zero when $Q\_i$ is learned to be 0. The penalty is that we must have $Q\_j >P\_j$ for at least one other element $j$ by the pigeonhole principle. In contrast, the forward KL has terms $P\_i \log \frac{P\_i}{Q\_i}$: when learning $Q\_i$, the only way to reduce the forward KL is for $Q\_i$ to more closely match $P\_i$. Thus, compared to the forward KL, the reverse KL may have a stronger tendency to ignore some modes.

However, the critical difference between continuous and discrete spaces may be the flexibility of the approximating distribution $Q$. In continuous spaces, "easy" choices of $Q$'s parametrization such as the normal distribution can often fail to include $P$ in real machine learning tasks: it is not possible to match $Q(x)=P(x)$ everywhere. More complicated parametrizations of $Q$, e.g., as a normalizing flow, can be required to make the family of $Q$ flexible enough to include $P$.

In discrete spaces with $k$ dimensions, the "easy" choice of $Q$ as a multinomial distribution with $k$ independent parameters, is fully flexible to specify any discrete distribution: $Q$ is powerful enough to match $P$ everywhere, since it can describe any point in the $k$-dimensional simplex. Since achieving $P=Q$ is generally accessible in discrete spaces, the relevance of mode-seeking or mode-covering descriptors (which can only describe properties of $Q$ when it doesn't match $P$) may be less meaningful. 

## Final notes

This topic has been explored many times before from several angles. Ben Moran had a nice [blog post](https://benmoran.wordpress.com/2012/07/11/distances-divergences-dirichlet-distributions/) on distances and distributions, relating L2 distance to Gaussian likelihood, L1 distance to the Laplace Distribution, and KL divergence on the simplex to Dirichlet likelihood. [Ben Lansdell](http://benlansdell.github.io/statistics/likelihood/) also derived a relationship between multinomial MLE and forward KL. In the context of document ranking for information retrieval, relationships between KL, multinomial, and dirichlet distributions was noted by [Nallapati et al.](https://www.researchgate.net/publication/228829646\_The\_Smoothed-Dirichlet\_distribution\_Explaining\_KL-divergence\_based\_ranking\_in\_Information\_Retrieval) in 2008.

# Summary

|                | Forward KL:  $\arg\min\_Q D\_{\texttt{KL}}(P \|Q)$             | Reverse KL:  $\arg\min\_Q D\_{\texttt{KL}}(Q \|P)$             |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Continuous** | Mode-covering, inclusive. $Q$ covers all of $P$'s modes | Mode-seeking, exclusive. $Q$ covers one of $P$'s modes. |
| **Discrete**   | $\propto -\log \textrm{MultinomialPMF}(\frac{\mathbf{x}}{n}=P; \mathbf{p}=Q) + c$ $=-\log \textrm{DirichletPDF}(x=Q; \alpha=P+1)+c$ $=-\log \textrm{DirichletPDF}(x=Q; \alpha=P)-KD\_{\texttt{KL}}(U\|Q)+c$ | Nothing of note                                              |

