
# Using multi-task learning to explain deep NLP models
[Tal Yifat](https://www.linkedin.com/in/tal-yifat/) & [‪Avi Segal‬](https://scholar.google.com/citations?hl=en&user=dYBHOY8AAAAJ)
## Overview 
### The research problem
Explainability has been a big research topic in recent years and saw the development several approaches to explaining deep learning models. While such methods have helped researchers gain insight into deep NLP models, these methods tend to be inaccessible to business users. The purpose of this research project is using [multi-task-learning](https://arxiv.org/abs/1706.05098) to develop explainability methods that would be accessible to and incorporate the input of business users.
### The data
The research uses [a public dataset on mine accidents from the U.S. Mine Safety and Health Administration](https://www.msha.gov/data-and-reports/mine-data-retrieval-system). The data includes a text narrative of the accident and structured data on the accident and its outcomes. The main task of the developed models is to predict, based on the narrative alone, whether a worker will lose at least 90 days of work as a result of the accident. Sample narratives:
* “Walking to electrical building to get supplies, slipped on ice and fell on left elbow  
causing a bruised elbow.”
* “Employee was pulling dust hose back, He was warned to watch out for the structure  
laying against the rib. He tripped over structure and fell on his back.”
* “Employee was knocking out block and block fell hitting employee in hand causing injury.”
### The business problem
The research project targets hypothetical business users who process disability insurance claims. For disability insurers, a large proportion of the cost comes from a relatively small number of claims in which workers take a long time to return to work. Identifying early on cases that have a high risk of turning into long-term disability claims could allow early intervention that would help customers recover more quickly and save money to the company. The project tests the hypothesis that textual data could help with early detection of such cases.  
## The inaccessibility of Deep NLP Explainability Methods
Let's review some of the main explainability method for deep NLP models (see some useful reviews: [1](https://arxiv.org/abs/2210.06929), [2](https://arxiv.org/abs/2010.00711) and [3](https://arxiv.org/abs/2108.04840)).
### Explaining knowledge in embeddings
Knowledge in embeddings is typically made explainable by enforcing the sparsity of the embeddings, either through methods that are intrinsic to the model or by transforming model embeddings into a comparable but interpretable space. Embedding dimensions are then interpreted by looking at the words/tokens that get extreme values in them, as in the example below, from Murphy et al., 2012](https://www.semanticscholar.org/paper/Learning-Effective-and-Interpretable-Semantic-using-Murphy-Talukdar/0048d3c3b41cdcc16dbe6fad545030dbed9722c6)
### Explaining knowledge in models' hidden layers
Transformer-based models offers the opportunity to exploit the attention mechanism and trace the how attention in specific neurons is allocated across inputs. Like knowledge in embeddings, this is interesting for NLP researchers, but inaccessible for business users. 
### Explaining model decisions through word attribution
Here, we get a little closer to business-oriented explainability methods, with techniques that attribute a model's decision to specific words/tokens from the input. The examples below, generated here with our data, demonstrate three such common methods. [SHAP](https://arxiv.org/abs/1705.07874) and [LIME](https://arxiv.org/abs/1602.04938) use manipulation and permutation of the inputs to infer their impact on model decisions. 

Another method, [Integrated Gradients](https://arxiv.org/abs/1703.01365), attributes the contribution of input features to a model's decision by examining the network gradients and comparing them to those a neutral baseline.  

We can see that attributing prediction impact to specific words is helpful, but not nearly as effective as a meaningful explanation we would get from a human expert, which would refer to higher structures of meaning that specific words or tokens. For example, such an expert explanation could be something like: "injuries in which employees fall from tall structures and hurt their back tend to require longer recovery time". 


> Written with [StackEdit](https://stackedit.io/).
