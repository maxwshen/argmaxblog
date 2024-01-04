---
 title: "Implicit Differentiation through Equilibria"
 math: mathjax
 author: ["Max Shen", "Jan-Christian Huetter"]
 summary: An introduction.
 date: 2022-07-20
---
*Max's note*: I've been pretty interested in deep implicit layers ([http://implicit-layers-tutorial.org/](http://implicit-layers-tutorial.org/), but I've found a lot of the online resources and papers describing implicit differentiation to have unsatisfying notation, probably because my calculus is more rusty than other people's. This post is my attempt to rewrite notes so that I can understand them more clearly.

# Implicit differentiation through equilibria

Consider a non-linear difference equation with
$$
\begin{aligned}
\mathbf{h}\_0 &= \mathbf{0} \\\ 
\mathbf{h}\_{t+1} &= f(\mathbf{h}\_t, \mathbf{x}, \boldsymbol{\theta}) \\\
\mathbf{h}\_{\texttt{ss}} &= f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) \\\
\mathbf{y}\_{\texttt{pred}} &= g(\mathbf{h}\_{\texttt{ss}}) \\\
\end{aligned}
$$
for input $x$, output $\mathbf{y}\_{\texttt{pred}}$, and parameters $\boldsymbol{\theta}$, and loss function $\ell(\mathbf{y}\_{\texttt{pred}})$, where $\texttt{ss}$ subscript stands for "steady state". This system can be viewed as an infinite-depth recurrent neural network where $\mathbf{h}$ is the hidden state, or as an infinitely deep neural network where every layer shares weights, or as a [deep equilibrium model](http://implicit-layers-tutorial.org/deep\_equilibrium\_models/). 

In this system, the input-output relationship $\mathbf{x} \rightarrow \mathbf{y}\_{\texttt{pred}}$ can be viewed as a function that is implicitly defined by $f$.

# Computing the steady state

First, this system is not guaranteed to have steady states, or they might not reachable. A sufficient (but not strictly necessary) criteria for the existence of steady state is that $f$ is *contractive*, which can depend on $\theta$. 

**Contractivity**: Consider a function $f: \mathbb{R}^d \rightarrow \mathbb{R}^d$ and a distance metric $d$ on $\mathbb{R}^d$. Then $f$ is contractive if $d(f(x), f(y)) < d(x, y)$ for all $x, y \in \mathbb{R}^d$. (Here $\mathbb{R}^d$ is our specific setting of interest, but this definition can be extended to other spaces.)

In our case, because $f$ also depends on $x$, it is possible that even if steady states exist for all $x$ in a training dataset $(x, y)$, we are not guaranteed that steady states will exist for new $x$ in the test set. However, work on deep equilibrium models suggests that when $f$ is a typical neural network layer, most values of $\boldsymbol{\theta}$ seem to support the existence of steady states.

To compute $\mathbf{h}\_{\texttt{ss}}$, the easiest approach is to employ *fixed point iteration*: initialize $\mathbf{h}\_0$ to the zero vector $\mathbf{0}$, then iteratively apply $f$ until $\mathbf{h}\_t \approx \mathbf{h}\_{t+1}$. This might take hundreds or thousands of iterations. This approach is intuitive; if we think of $f$ as a neural network, it's natural to compute a forward pass through $T$ time steps like an RNN. If we run this forward pass with autograd listening, it's easy to backpropagate to learn $\boldsymbol{\theta}$. 

However, backpropagation through time is not a desirable solution here. If we have $T$ unrolled steps and $P$ parameters, then we require $O(TP)$ space to store the full computation graph, and $O(TP)$ time to compute it!

It turns out that using implicit differentiation, we can do much better. Specifically, implicit differentiation gives us as way to compute the gradient *without needing* the fully unrolled computation graph - it only needs the value of the fixed point $\mathbf{h}\_{\texttt{ss}}$, and therefore only requires $O(P)$ space. This means that we can choose *any* approach to finding the steady state, and we *don't care* about the trajectory we used to reach the steady state. This opens the door to using more sophisticated fixed point solvers, like Newton's method or Anderson acceleration: we only need to store the final steady state, and can avoid storing the trajectory. 

# Learning the parameters $\boldsymbol{\theta}$

To learn with our loss function, we have

$$
\frac{
   d \ell(\mathbf{y}\_{\texttt{pred}})}{ d \boldsymbol{\theta}} = 
\underbrace{
    \frac{ d \ell(\mathbf{y}\_{\texttt{pred}}) }
		{ d \mathbf{y}\_{\texttt{pred}}} 
    \frac{ d \mathbf{y}\_{\texttt{pred}}}
		{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta})}
}\_{\textrm{accessible}}
	\textcolor{red}{
		\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta})}{ d \boldsymbol{\theta}}
        }
$$

where we write $\mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta})$ to explicitly describe the dependence of $\mathbf{h}\_{\texttt{ss}}$ on $\boldsymbol{\theta}$ (that is, if $\boldsymbol{\theta}$ changes, $\mathbf{h}\_{\texttt{ss}}$ also changes). The black right-side terms are accessible from autograd, while the red term is the tricky part.

Recall that $\mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) = f(\mathbf{h}\_{ss}(\boldsymbol{\theta}), \mathbf{x}, \boldsymbol{\theta})$. When computing the *total derivative* $\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}$, recall that $f(\mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}), \mathbf{x}, \boldsymbol{\theta})$ depends on $\boldsymbol{\theta}$ in two ways: indirectly through $\mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta})$ and directly through $\boldsymbol{\theta}$. Using the multivariate chain rule of partial derivatives, we have:

$$
\begin{aligned}
\textcolor{red}{
	\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta})}{ d \boldsymbol{\theta}}} 
&= \frac{ d f(\mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}), \mathbf{x}, \boldsymbol{\theta})}{ d \boldsymbol{\theta}} \\\
&= 
\underbrace{
	\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }
		 { \partial 	\mathbf{h}\_{\texttt{ss}}}
    \underbrace{
		\textcolor{red}{
        \frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }
			 { d \boldsymbol{\theta}}
			 }
    }\_{\textrm{Same as left side}
    }
}
\_{ \textrm{differentiation through } \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }
+ 
\underbrace{ \frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \boldsymbol{\theta} } }\_{\textrm{through }\boldsymbol{\theta} }
\end{aligned}
$$

where we differentiate with respect to $\boldsymbol{\theta}$ through $\mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta})$ (left term in the sum) and directly through $\boldsymbol{\theta}$ (right term in the sum). Note that in the last right-side partial derivative term, $\mathbf{h}\_{\texttt{ss}}$ is held constant when differentiating with respect to $\boldsymbol{\theta}$. 

We end up with another recurrence relation, where our goal is to solve for $\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}$. Note that the black terms are easily computed with autograd when we have the fixed point $\mathbf{h}\_{\texttt{ss}}$. Simply call the function $f$ with inputs  $\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}$ with autograd listening.

$$
\begin{aligned}
\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}
&= 
\underbrace{
	\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \mathbf{h}\_{\texttt{ss}}}
}\_{\textrm{from autograd}}
\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}
+ 
\underbrace{
	\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \boldsymbol{\theta} }
}\_{\textrm{ from autograd } }
\end{aligned}
$$

This equation represents our main challenge, and in the next sections we will discuss several strategies for solving it.

**A practical note**: Recall that for a function $f: \mathbb{R}^{m} \rightarrow \mathbb{R}^n$, reverse-mode differentiation requires $O(n)$ time, while forward-mode differentiation takes $O(m)$ time. In practice, standard backpropagation with autograd libraries use reverse-mode differentiation, as many deep learning loss functions are scalars. If we say that computing the gradient w.r.t. a 1-D output requires 1 backprop pass, then computing the Jacobian $\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \mathbf{h}\_{\texttt{ss}}}$ with $d$ dimensions requires $d$ backprop passes. For exposition, it's easiest to describe implicit differentiation with the object $\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \mathbf{h}\_{\texttt{ss}}}$, but in practice one should compute the fixed point iteration on  $\frac{ d \ell(\mathbf{y}\_{\texttt{pred}})}{ d \boldsymbol{\theta}}$ instead.

**A convenient property:**  It turns out that if $f$ is contractive, then $\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \mathbf{h}\_{\texttt{ss}}}$ is contractive as well, so the existence and reachability of a fixed point for $\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}$ is achievable.

## Strategy 1: Fixed point iteration

To solve for $\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}$ if we have the black terms, we can initialize $\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}$ to any arbitrary value, then iterate the equation until convergence. This is analogous to solving for $\mathbf{h}\_{\texttt{ss}}$ in the forward pass with fixed point iteration by iteratively applying the function. 

As with the forward pass, we can employ more sophisticated fixed point solvers, like Newton's methods or Anderson acceleration.

## Strategy 2: Neumann series approximation

We can rewrite:

$$
\begin{aligned}
\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}
&= 
\underbrace{
	\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \mathbf{h}\_{\texttt{ss}}}
}\_{\textrm{from autograd}}
\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}
+ 
\underbrace{
	\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \boldsymbol{\theta} }
}\_{\textrm{ from autograd } }
\\\
\Bigg( \mathbf{I}
\-
\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \mathbf{h}\_{\texttt{ss}}}
\Bigg)
\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}
&= 
\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \boldsymbol{\theta} }
\\\
\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}( \boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}
&= 
\underbrace{
    \Bigg( \mathbf{I}
    -
    \underbrace{
        \frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \mathbf{h}\_{\texttt{ss}}}
    }\_{\textrm{ from autograd }}
    \Bigg
    )^{-1}
}\_{\textrm{need to compute inverse}}
\underbrace{
	\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \boldsymbol{\theta} }
}\_{\textrm{1 autograd pass}}
\end{aligned}
$$

which can be understood as trying to directly solve a linear system of equations with matrix inversion. In contrast to fixed point iteration which might take many iteration steps, if we had $\Big( \mathbf{I} - \frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \mathbf{h}\_{\texttt{ss}}}
    \Big
    )^{-1}$, we could solve for $\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}$ in a single step. The main challenge then becomes computing the inverse.

The Neumann series states that for any invertible matrix $\mathbf{A}$,
$$
\Big(\mathbf{I} - \mathbf{A} \Big)^{-1} = \sum\_{k=0}^{\infty} \mathbf{A}^k.
$$
This directly gives us a strategy to solve for $\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}$. 

```python
def get_red_grad_approx(grad_h, grad_theta, neumann_inv_iter=10):
    # grad_h is df/dhss, grad_theta is df/dtheta
    neumann_approx = sum([grad_h**k for k in range(neumann_inv_iter)])
    return neumann_approx @ grad_theta
```

**In practice, strategies 1 and 2 are equivalent.** It turns out that for a finite number of iterations $k$, this Neumann series is exactly equivalent to $k$ fixed-point iterations when we initialize our guess at $\mathbf{0}$. For clarity since this point applies in general, let's use simpler notation. Our iteration is on $\mathbf{x}$ is:
$$
\mathbf{x = Ax + b} \\\
$$
So iterating a few times, we have:
$$
\begin{aligned}
\mathbf{A(0) + b} &= \mathbf{b} \\\
\mathbf{A(b)+b} &= \mathbf{(A+I)b} \\\
\mathbf{A(Ab+b)+b} &= \mathbf{(A^2 + A+I)b} \\\
\end{aligned}
$$
Our Neumann series tells us that
$$
\begin{aligned}
\mathbf{x = (I-A)^{-1} \mathbf{b} } \\\
\mathbf{x} = \Big(\sum\_{k=0}^{\infty} \mathbf{A}^k \Big) \mathbf{b} \\\
\end{aligned}
$$
 which has the same form.

## How low can you go?

The zero-th order Neumann approximation is interesting:
$$
\Big(\mathbf{I} - \mathbf{A} \Big)^{-1} \approx \mathbf{A}^0 = \mathbf{I}
$$
This seems ridiculous! But surprisingly, there's a theoretical argument ([Wu Fung et al., 2021](https://arxiv.org/pdf/2103.12803.pdf)) that using this approximation in the gradient:

$$
\begin{aligned}
\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}
&=
    \underbrace{
    \Bigg( \mathbf{I}
    \-
    \underbrace{
        \frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \mathbf{h}\_{\texttt{ss}}}
    }\_{\textrm{ from autograd }}
    \Bigg
    )^{-1}
}\_{\textrm{need to compute inverse}}
\underbrace{
	\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \boldsymbol{\theta} }
}\_{\textrm{ from autograd }} \\\
&\approx
\mathbf{I} \underbrace{
	\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \boldsymbol{\theta} }
}\_{\textrm{ from autograd }} 
\\\
&\approx
 \underbrace{
	\frac{ \partial f(\mathbf{h}\_{\texttt{ss}}, \mathbf{x}, \boldsymbol{\theta}) }{ \partial \boldsymbol{\theta} }
}\_{\textrm{ from autograd }} 
\end{aligned}
$$

yields a $\textcolor{red}{\frac{ d \mathbf{h}\_{\texttt{ss}}(\boldsymbol{\theta}) }{ d \boldsymbol{\theta}}}$ that is a valid descent direction. Sometimes being cheap can pay off!

What does this approximation mean? Even though $f(\mathbf{h}\_{\texttt{ss}}(\theta), \mathbf{x}, \theta)$ depends on $\boldsymbol{\theta}$ indirectly through the recurrence relation *and* directly through $\theta$, it tells us to ignore the dependence through the recurrence relation and only consider the direct dependence.

Maybe intuitively this makes sense: if $f$ is "stable" enough such that changing $\boldsymbol{\theta}$ a little bit will still get us to a steady state, then the value of the steady state doesn't depend on how $\boldsymbol{\theta}$ changes the trajectory we used to get there. Instead, changing $\boldsymbol{\theta}$ will only change the value of the steady state according to the local curvature of how the steady state depends on $\boldsymbol{\theta}$ immediately around the steady state. 

```python
def get_red_grad_approx(grad_h, grad_theta):
    # grad_h is df/dhss, grad_theta is df/dtheta
    return grad_theta
```

Wu fung et al. report a 60%-400% speedup and similar test-set performance with this approach compared to using $k=5$ Neumann iterations.



# Time and space complexity

Suppose we have $P$ parameters.

|                                                              | Time    | Space   |
| ------------------------------------------------------------ | ------- | ------- |
| Backpropagation through time (with $T$ time steps)           | $O(TP)$ | $O(TP)$ |
| Implicit differentiation: $k$ fixed-point iterations, or Neumann steps | $O(kP)$ | $O(P)$  |
| Implicit differentiation: zero-th order Neumann              | $O(P)$  | $O(P)$  |

