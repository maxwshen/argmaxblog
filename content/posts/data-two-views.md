---
  title: "What is data? The classical and postmodern views"
  math: mathjax / katex
  summary: Is data sacred?
  date: 2024-03-27
---
<!-- # What is data? The classical and postmodern views -->

Data is the oil that powers machine learning. But beyond image and text domains where data is abundant, many important problems are relatively data poor. Even many ML applications on images and text can suffer from data scarcity, when these tasks care about data in the long tail of Zipfian data distributions. This problem has motivated interest in *synthetic data*, which seems surprisingly controversial to many: see [Listgarten's perspective "The perpetual motion machine of AI-generated data and the distraction of ChatGPT as a ‘scientist’"](https://www.nature.com/articles/s41587-023-02103-0). 

To understand machine learning, one should have an understanding of what data is. Suppose you are given a long string of binary digits by an unknown entity. Would you be able to ask a series of questions to this entity, to say with confidence whether you "have data" or not?

Here, I share two views on the question: *what is data?* 
<!-- The classical and postmodern views hold different things to be sacred, which may inflict religious-like fervor on devotees of each camp. -->

**The classical view** holds that data is a measurement of objective reality. Data is sacred - it is something we obtain by "venturing out" and spending resources to query *nature* or *reality*, and data is how nature responds to our query. To "make up" synthetic data is profane, equivalent to humans daring to play God. This view is more common among old-school statisticians.

**The postmodern view** views data simply as an ingredient used in the process of training a machine learning model, among other ingredients like choosing the architecture, loss function, and other hyperparameters. If you change the architecture and run the training process, you get out a model with different behavior. Similarly, the postmodern view understands data as a toggle or lever: changing the data will change the behavior of the resulting model. This is purely a *functional* view of data (what can it do for me?); in this framework, "is the data real" is not relevant or perhaps even sensible.

This view is more common among modern deep learning practitioners, and echoes a conceptual schism that occurred in old-school statistics: ["Two Cultures of Statistical Modeling" by Leo Breiman (2001)](https://www2.math.uu.se/~thulin/mm/breiman.pdf): if black-box random forests (an ugly model, to perhaps many statisticians at the time) achieve better test-set predictive performance, then favor them over elegant statistical models hand-crafted with expert knowledge. 

In the postmodern view, we primarily care about improving the behavior of the model. Thus, *evaluation* becomes sacred, as the main force grounding us in reality. The postmodern view would embrace synthetic data if it improves performance in evaluation.

Notably, this {data $\rightarrow$ behavior} framework has a non-trivial relationship with generalization. Generalization can be irrelevant to the desired behavior on tasks where memorization can nearly be enough, e.g.  [solvable through advanced retrieval techniques](https://towardsdatascience.com/the-road-to-biology-2-0-will-pass-through-black-box-data-bbd00fabf959). In this situation, we can abandon even the "sacred" division of data into training set and held-out test-set, and simply put everything into the training set. A concrete example is using an LLM to simply look up information: the best way to maximize performance is to put everything into the training set.

Taking the concept of synthetic data to an extreme, the postmodern view suggests an intriguing problem: to maximize the performance of a model: search over the data space for the "optimal dataset". In general, this is an incredibly challenging problem in a very high-dimensional search space, but this is exactly the the type of setting where deep learning is commonly employed.

**Reconciliation.** How shall one reconcile these two views? I favor having an understanding of each perspective in one's toolbox, and setting sacredness aside. It is a fact that data *is* an ingredient that, if modified, can change model behavior, and therefore that synthetic data *can* improve model performance. However, "real" data is a key tether to reality, and sacrificing that tether can make trusting a model harder, by increasing the burden of proving trust on evaluation. 