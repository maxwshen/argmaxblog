---
    title: "Sampler evaluation: No mode-covering without importance sampling"
    math: mathjax
    summary: An impossibility result 
    date: 2025-09-17
---
<!-- # Sampler evaluation: No mode-covering without importance sampling -->

In a previous blog post, we pinned down a definition for mode-covering vs. mode-seeking behavior for distributional divergences. When fitting $q$ to a target $p$, we studied the derivative norm of the divergence wrt $q(x)$ where $p(x) > 0$ and $q(x) \to 0$. 

> *Mode-covering pressure* means that when $p(x) > 0$ and $q(x) \to 0$, the gradient of the divergence w.r.t. $q(x)$ diverges strongly (at least $O(p/q)$), pushing $q$ to cover that region.

The forward KL has $O(p/q)$ mode-covering pressure, which is exponentially stronger than the reverse KL with  $O(\log(p/q))$ pressure.

These divergence behaviors matter not only for training, but also for evaluating samplers: if our metric is mode-seeking, it can overstate quality by ignoring missing modes.

# Sampler evaluation

Now, let us consider the problem of sampler evaluation. This entails constructing evaluation metrics for the quality of a sampler. First, we'll have to define what the sampling problem is, and the goal.

The sampling problem is: given a target distribution $p(x)$ known only up to its unnormalized density $f(x)$ with unknown normalizing constant $Z$, we wish to obtain iid samples $\{x_i\}_{i=1}^N \sim p(x)$.

There are many types of approaches for sampling, but we'll focus on learning likelihood-based samplers, which includes neural samplers like normalizing flows, autoregressive models, and/or diffusion models. These aim to learn a model distribution $q(x)$ that matches $p(x)$.

We presume that we can both sample from $q$ and evaluate model likelihoods $q(x)$. In contrast, we can evaluate the unnormalized density $f$ of $p$, and cannot sample from $p$.

With these access conditions, we can evaluate the reverse KL. For simplicity of exposition we will write it with $p$; in practice we use $f$ instead, and incur an unknown scaling factor related to $Z$.

$$
\text{Reverse KL}: \quad KL(q||p) = \int q(x) \log\frac{q(x)}{p(x)} dx
$$

However, a key challenge in high-dimensional sampling problems is discovering and covering all modes of $p$.
We are not interested in the reverse KL because it is mode-seeking, not mode-covering.

# Mode-covering sampler evaluation metrics
We could try estimating the forward KL, but this requires importance sampling which works poorly in high dimensions, especially when $q$ is very different from $p$.

$$
\begin{align}
\text{Forward KL}: \quad KL(p||q) &= \int p(x) \log\frac{p(x)}{q(x)} dx \\\
&= \int q(x) \frac{p(x)}{q(x)} \log\frac{p(x)}{q(x)} dx \\\
&= \mathbb{E}\_q \left[ \underbrace{\frac{p(x)}{q(x)}}\_{\text{Importance weight}} \log\frac{p(x)}{q(x)} \right]
\end{align}
$$


## Impossibility result
This motivates asking: is there a mode-covering metric that does not use importance sampling?

Unfortunately, satisfying these two properties is provably impossible:

1. No importance sampling, when estimating the metric as an expectation under $q$, meaning no importance weight factors like $p(x)/q(x)$.
2. $O(p/q)$ mode-covering pressure, like forward KL

## Proof

The proof follows by understanding where the $O(p/q)$ mode-covering pressure comes from mathematically in the forward KL definition. In short, it comes from the term $p \log (q)$ in the integrand, because:

$$
\frac{\partial}{\partial q} p\log(q) = p \frac{1}{q} = O(p/q)
$$

Thus, it is necessary to have $p \log (q)$ in the integrand to achieve $O(p/q)$ mode-covering pressure.

In contrast, consider what happens when we estimate a metric as an expectation under $q$: the integrand is multiplied by a factor $q(x)$. If we have a term $q \log(q)$ in the integrand, its derivative by the product rule becomes:

$$
\frac{\partial}{\partial q} q \log(q) = q\frac{1}{q} + 1 \log(1/q) = O(\log(1/q)
$$

Having an expectation w.r.t. $q$ significantly dampens the mode-covering pressure by the product rule. We can see that the desired $1/q$ term, which is the gradient of $\log(q)$ is canceled by the front term $q$. The only way to keep the forward KL-like pressure is to importance weight by $p/q$.

This completes the proof.

---

# Sidenote: Bhattacharyya / Amari / Rényi divergences

A follow-on question is: so far, we have considered $O(p/q)$ as the bar for mode-covering, but there's an exponential gap between $O(p/q)$ and $O(\log(p/q))$. Could there be intermediates that are "mode-covering"?

Indeed, the Bhattacharyya coefficient is another measure of distributional similarity, and is the "middle ground" between reverse and forward KL in the Amari divergence family. 

$$
\text{Bhattacharyya coefficient}:\quad BC(p, q) = \int \sqrt{p(x)q(x)} dx = \mathbb{E}_q \left[ \sqrt{\frac{p(x)}{q(x)}} \right]
$$

The BC has favorable properties: it has bounded, finite variance, is stable to estimate as an expectation of $q$, and has mode-covering pressure $O(\sqrt{p/q})$.

However, in an empirical evaluation of its mode-covering behavior, I found that it behaves more similarly to the reverse KL in being mode-seeking, and not similarly to the forward KL in being mode-covering. Specifically, I trained ten normalizing flow (TarFlow) models on MNIST on the first $k$ digits. I used the model trained on all 10 digits as the target $p$, and used other models trained on fewer digits as $q$, simulating dropping modes.
This example captures a lot of real-world complexity: digits vary in how easy they are to model, so the neural sampler likelihoods vary significantly by digits. At the same time, I verified that neural samplers trained on digit subsets reliably sample high-quality images, and it is reasonable to consider the ten digits as ten separate modes.

The true forward KL, estimated with samples from $p$, correctly scores the models in terms of how many modes of $p$ they miss. The reverse KL fails to retrieve this ordering, and so does the Bhattacharyya coefficient.

---

# Conclusion

In the last blog post of this series, we'll take a look at Stein discrepancy, which is a different way to measure distributional similarity that is often considered in the sampler evaluation setting, because it does not require iid samples from the target $p$, and is compatible with knowing $p$ only up to an unnormalized density. We will show that the Stein discrepancy is mode-seeking.