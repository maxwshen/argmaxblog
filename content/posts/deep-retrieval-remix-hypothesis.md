---
  title: "The deep retrieval + remixing hypothesis"
  math: mathjax
  summary: How do modern deep generative models produce high-quality outputs?
  date: 2024-03-28
---

A common perspective on modern deep models focuses on their surprising capabilities for memorization. François Chollet has [tweeted](https://twitter.com/fchollet/status/1755250582334709970):

> LLMs struggle with generalization (the only thing that actually matters), due to being entirely reliant on memorization

[and](https://twitter.com/fchollet/status/1755270681359716611)

> LLMs = 100% memorization. There is no other mechanism at work. A LLM is a curve fitted to a dataset (that is to say, a memory), with a sampling mechanism on top (which uses a RNG, so it can generate never-seen-before token sequences). It doesn't just memorize and output back exact token sequences, obviously. It can memorize and reuse any pattern found in its training data, include certain categories of programs. But crucially, those patterns must have been featured in the training data. They need to have been memorized. LLMs break down on anything that wasn't in the their training data. Like ARC. Because they're 100% memorization. Which makes sense, because a LLM is literally a curve fitted to some data points. What else could you expect?

A similar idea is posited in [The Road to Biology 2.0 Will Pass Through Black-Box Data](https://towardsdatascience.com/the-road-to-biology-2-0-will-pass-through-black-box-data-bbd00fabf959) (Bronstein, Naef) that "degenerate solution spaces" underlie the performance of AlphaFold and LLMs, where valid solutions can be found by remixing retrieved subparts from the training data. (Emphasis my own):

> **“Degenerate” solution space.** Another peculiarity of AlphaFold2 is that the supervised training set of only 140K protein structures and 350K sequences and is tiny by ML standards [29] — an order of magnitude less than the amount of data used to train AlexNet almost a decade earlier, and a drop in the ocean compared to the contemporaneous GPT-3 [18]. What likely makes such a small dataset sufficient is the “degeneracy” of the solution space: while in theory the number of all possible solutions in protein folding is astronomically large (estimated at 10³⁰⁰ [30]), only a very small fraction thereof is actualised. This is akin to the “manifold hypothesis” in computer vision, stating that natural images form a low-dimensional subspace in the space of all possible pixel colours [31].
> 
> The reason for this “degeneracy” likely lies with evolution: most of the proteins we know have emerged over 3.5 billion years of evolutionary optimisation in which existing domains were copied, pasted, and mutated [32], **producing a limited “vocabulary” that is reused over and over again.** There are thermodynamic reasons for this, too, as only a limited set of possible amino acid 3D arrangements make up for the entropic cost of a defined protein fold [33]. **Most protein folds can thus be achieved by recombining and slightly modifying existing ones and valid solutions can be formed through advanced retrieval techniques** [34].
>
> From this perspective, the protein folding problem is reminiscent of natural language, where tasks such as writing, coding, translation, and conversation where LLMs excel and which often do not require strong generalisation and **can be solved by recombining existing examples** [35] (e.g. copy-pasting pieces of code from GitHub code repositories that GPT-4 has been trained on).
>
> Thus, while AlphaFold2 has worked remarkably well in predicting the structures of proteins never crystallised before, it might be not due to its generalisation capabilities but on the contrary, because it *does not need to generalise* [36]. An observation in favour of this hypothesis is that even the most recent update to AlphaFold (AlphaFold-latest, which introduces small molecules), and other state-of-the-art docking algorithms appear to struggle in generalising to previously unseen protein-ligand complexes [37–38], as unlike proteins most small molecules do not evolve, and therefore the solution space is likely less degenerate and less well reflected in the training data [39–40].

I'm not the biggest fan of the phrase "degenerate solution space" -- it is not clear what particular solution space is referred to: for AlphaFold, the solution space seems to be implicitly defined by the training dataset and the loss function. AlphaFold performs well on full-length, wild-type protein sequences, but is known to perform poorly when introducing single mutations, or separately, large deletions, which are outside the "solution space" that AlphaFold was trained to care about. The issue with the phrase "degenerate solution space" is there is not just one, but multiple solution spaces we care about, and a model can perform well in one space but poorly in another.

My preferred framing focuses on a functional process: the *deep retrieval & remixing hypothesis*:

{{< box info >}}
**Deep retrieval & remixing hypothesis**

Modern deep generative models primarily generate high-quality outputs by remixing retrieved subparts from the training data.
{{< /box >}}

This is often taken to be a criticism, suggesting performance is illusory or not to be taken seriously. Instead of embracing it or rejecting it outright, let's hold it at arms length, investigate its implications, and see what it predicts and how it can be tested.

## Corollary 1: Modern deep generative models perform best on problem classes where good solutions are achievable by remixing retrieved elements.
Beyond protein folding and some (but not all) text generation applications as covered by Bronstein and Naef, it seems to me that text-to-image generation and video generation are also problem classes where remixing seen entities can produce solutions that humans are happy with: e.g., "astronaut riding a horse on mars." [Sora from OpenAI](https://openai.com/sora) seems best at generating videos in a drone style, with shallow depth of field (blurred background), or Unity game engine style, but appears capable of combining known entities: e.g., paper airplanes flying over a forest, or [a man with a yellow balloon as a head](https://openai.com/blog/sora-first-impressions).

LLMs are particularly useful for generating code using a particular library or API (like matplotlib) with plentiful examples in the training data, where the human user doesn't have the library memorized and thus needs relatively common snippets of code. This task appears especially solvable with deep retrieval and remixing.

Furthermore, models seem to have a bias towards remixing retrieved elements, even when the text prompt or input strays away from the typical. AlphaFold is "insensitive to isoforms with large deletions", predicting a structure as though the deletion didn't occur. AlphaFold also underestimates the effect of mutations which can drastically change structure. 

My overall rating: Anecdotal observations like these, and abundant reports of the surprising memorization capabilities of deep models, across multiple domains suggests that the deep retrieval and remixing hypothesis is plausible.

A testable prediction implied by the deep retrieval & remixing hypothesis is that *music generation* is a domain that modern deep generative models will excel at. I expand on this implication in a separate blog post, where I discuss how music as a domain enjoys many favorable properties in contrast to text/image/video generation, suggesting that deep generative models can have even greater, outsized impact for music.

## Corollary 2: Modern deep generative models will perform poorly on tasks that can't be solved by remixing retrieved elements.

I find negative evidence for this corollary for LLMs in particular. LLMs are not purely retrieval and remixing machines, as they also seem to learn to execute algorithms to a certain degree. This is showcased by the in-context learning ability of LLMs, and the finding that [transformers can in-context execute algorithms like linear regression on unseen data](https://papers.nips.cc/paper\_files/paper/2022/hash/c529dba08a146ea8d6cf715ae8930cbe-Abstract-Conference.html). LLMs are capable of executing simple text manipulation instructions, like switching characters around or applying a cipher, to text that a user can ensure has never been seen during training, by randomly generating input text. LLMs are good at summarizing text, which is more like applying an "algorithm" than regurgitating memorized info. This performance is not explainable as just remixing elements retrieved from training data.

Thus for LLMs in particular, I posit a revised hypothesis:

{{< box important >}}
LLMs employ deep retrieval & remixing as one mechanism, among others.
{{< /box >}}

This revised LLM hypothesis implies that LLMs should work well on tasks amenable to retrieval + remixing, but LLMs may also work well on tasks not amenable to it, by employing other mechanisms like executing learned algorithms.

For image/video generation or protein folding, I have not yet encountered analogous examples that would serve as negative evidence, but I would be very interested in this.

Sidenote: Chollet [sees that LLMs can learn to execute algorithms](https://twitter.com/fchollet/status/1689703571493896194), but understands this as another type of memorization, which is a rather fuzzy word. I use the term retrieval here, which leads me to the opposite conclusion as Chollet.

# How does the deep retrieval & remixing hypothesis relate to generalization?

Bronstein and Nauf write:

> Thus, while AlphaFold2 has worked remarkably well in predicting the structures of proteins never crystallised before [... by using deep retrieval and remixing ...], it might be not due to its generalisation capabilities but on the contrary, because it *does not need to generalise* [36]. 

In other words, Bronstein and Nauf write that using a deep retrieval and remixing mechanism means that generalization is no longer needed.

I disagree - I posit that:

{{< box info >}}
Claim: Deep retrieval & remixing can be a valid mechanism for achieving valid in-distribution generalization.
{{< /box >}}

In my view, the remixing abilities of modern deep generative models are the most mysterious and miraculous parts of them. Text-to-image models are able to stitch together retrieved elements in novel compositions or arrangements (e.g., "chair shaped like an avocado", or "astronaut riding a horse on Mars") in ways that often surprisingly natural and satisfying to humans.

Suppose a model $f$ has memorized solutions for $N$ subproblems $x\_1, ..., x\_N$, but we are interested in solving larger problems $y$, where each larger problem $y$ is a list of $K$ subproblems. Further, suppose we know an aggregation, or "remixing"/"combining" function $\phi$, where the solution for larger problem $y$ is $\phi(f(y\_1), ..., f(y\_K))$.

Then with $N$ memorized subproblem solutions and knowledge of $\phi$, we can solve $N^K$ larger problems - much larger than $N$.

Now, suppose we collected a training dataset that is a small fraction of all $N^k$ larger problems, and our model managed to memorize $N$ subproblem solutions and learn $\phi$: we could say that this model, by remixing or combining the retrieved memorized subproblem solutions, has achieved in-distribution generalization to the space of $N^K$ problems. I might argue that deep retrieval & remixing mechanisms may effectively achieve this scenario.

**Remixing abilities can be relatively simple, but surprisingly powerful**

Basic algorithms, like sorting or elementary-school pencil-and-paper addition, achieve perfect performance over their entire problem class; in ML parlance, they achieve perfect generalization -- not due to memorization, but primarily because the algorithm is *correct over the entire problem class*. In the notation above, we are correct over the $N^K$ space *if and only if* $\phi$ is correct over the $N^K$ space: we inherit its generalization power.

Consider pencil-and-paper addition: we can posit a memorized table of single-digit addition, i.e., we memorize that 1+1=2, 1+2=3, ..., 9+9=18. We then apply an algorithm $(\phi)$ which proceeds right-to-left where we add single digits, and if the result $\geq$ 10, we carry over a 1, and iterate. This procedure lets us solve an *infinite* number of addition problems, despite using a very small memorization table.

**Generalization doesn't have to be sexy**

It feels like generalization is treated as a mysterious, holy grail, magical promise of machine learning, and so it has to occur by a sophisticated, uninterpretable mechanism. And the community is somehow disappointed if it senses that a model's mechanics aren't high-brow enough: it isn't *generalizing*, it's *merely...*. 

Recall that the definition of generalization is that a model performs well (similar to its training performance) on new, unseen data drawn from the same data distribution it was trained on. AlphaFold performs well on never-before crystallized proteins; it generalizes, even if its hypothesized mechanism for achieving this is unsexy.

<!-- **Just semantic games?** -->

<!-- An unfortunate aspect of this disagreement is the nebulous nature of what is "in-distribution" versus "out-of-distribution" -- the data distribution is only implicitly implied in practice, and too underspecified for humans to ever come to consensus on what it is.  -->
<!-- Generalization is similarly nebulous, where "out-of-distribution generalization" is generally "hard".  -->
<!-- The subjective nature of these concepts unfortunately means no one can refute a claim that a model has failed because it was given out-of-distribution inputs. -->
<!-- The danger is that  -->