# Using multi-task learning to explain large Language models (LLMs)
[Tal Yifat](https://www.linkedin.com/in/tal-yifat/) & [‪Avi Segal‬](https://scholar.google.com/citations?hl=en&user=dYBHOY8AAAAJ)
## Overview 
### The research problem
Explainability has been a big research topic in recent years and saw the development several approaches to explaining deep learning models. While such methods have helped researchers gain insight into deep NLP models, these methods tend to be inaccessible to business users. The purpose of this research project is using [multi-task-learning](https://arxiv.org/abs/1706.05098) to develop explainability methods that would be accessible to and incorporate the input of business users.
### The data
The research uses [a public dataset on mine accidents from the U.S. Mine Safety and Health Administration](https://www.msha.gov/data-and-reports/mine-data-retrieval-system). The data includes a text narrative of the accident and structured data on the accident and its outcomes. The main task of the developed models is to predict, based on the narrative alone, whether a worker will lose at least 90 days of work as a result of the accident. 10% of the data points had positive labels. Sample narratives:
* “Walking to electrical building to get supplies, slipped on ice and fell on left elbow  
causing a bruised elbow.”
* “Employee was pulling dust hose back, He was warned to watch out for the structure  
laying against the rib. He tripped over structure and fell on his back.”
* “Employee was knocking out block and block fell hitting employee in hand causing injury.”
### The business problem
The research project targets hypothetical business users who process disability insurance claims. For disability insurers, a large proportion of the cost comes from a relatively small number of claims in which workers take a long time to return to work. Identifying early on cases that have a high risk of turning into long-term disability claims could allow early intervention that would help customers recover more quickly and save money to the company. The project tests the hypothesis that textual data could help with early detection of such cases.  
### Current results and notebooks
The project is a work in process, and here are the notebooks published so far:
* [A notebook that trained Deberta V3 base on the primary task of classifying the accidents according to whether the worker lost at least 90 days of work](https://github.com/tal-yifat/deep-NLP-explainability/blob/main/Primary_task_detecting_long_term_work_absence.ipynb). The target metric was ROC-AUC and the model performance was 0.80. 
* [A notebook in which an auxiliary, explanatory task is trained by itself](https://github.com/tal-yifat/deep-NLP-explainability/blob/main/Auxilary_task_injury_explanation.ipynb) (to test the viability before using it in multi-task learning). The task was predicting the injured body part, among seven possible labels, based on the narrative alone. The model, Deberta V3 base, had ROC-AUC (macro average, one-over-rest) of 0.97 (a much easier task apparently!).
* [A notebook that demonstrates the use of established explainability techniques to the primary classification task - LIME, SHAP, and Integrated Gradients](https://github.com/tal-yifat/deep-NLP-explainability/blob/main/Common_NLP_explainability_methods.ipynb). 
## The inaccessibility of Deep NLP Explainability Methods
Let's review some of the main explainability method for deep NLP models (see some useful reviews: [1](https://arxiv.org/abs/2210.06929), [2](https://arxiv.org/abs/2010.00711) and [3](https://arxiv.org/abs/2108.04840)).
### Explaining knowledge in embeddings
Knowledge in embeddings is typically made explainable by enforcing the sparsity of the embeddings, either through methods that are intrinsic to the model or by transforming model embeddings into a comparable but interpretable space. Embedding dimensions are then interpreted by looking at the words/tokens that get extreme values in them, as in the example below, from [Murphy et al., 2012](https://www.semanticscholar.org/paper/Learning-Effective-and-Interpretable-Semantic-using-Murphy-Talukdar/0048d3c3b41cdcc16dbe6fad545030dbed9722c6)

![Example of interpretable embeddings](https://github.com/tal-yifat/deep-NLP-explainability/blob/main/Images/Embeddings.png)
### Explaining knowledge in models' hidden layers
Transformer-based models offers the opportunity to exploit the attention mechanism and trace the how attention in specific neurons is allocated across inputs. We can see how the attention in a specific layer is allocated when a certain word is processed. Like knowledge in embeddings, this is interesting for NLP researchers, but inaccessible for business users. We can see this in the following example from [exBERT.net](https://exbert.net/). 

![exBERT](https://github.com/tal-yifat/deep-NLP-explainability/blob/main/Images/Exbert.png)
### Explaining model decisions through word attribution
Here, we get a little closer to business-oriented explainability methods, with techniques that attribute a model's decision to specific words/tokens from the input. The examples below, generated here with our data, demonstrate three such common methods. [SHAP](https://arxiv.org/abs/1705.07874) and [LIME](https://arxiv.org/abs/1602.04938) use manipulation and permutation of the inputs to infer their impact on model decisions. We can see that words are highlighted according to their contribution to a positive or negative prediction.
#### SHAP
![SHAP](https://github.com/tal-yifat/deep-NLP-explainability/blob/main/Images/SHAP.png)
#### LIME
![LIME](https://github.com/tal-yifat/deep-NLP-explainability/blob/main/Images/LIME.png)
Another method, [Integrated Gradients](https://arxiv.org/abs/1703.01365), attributes the contribution of input features to a model's decision by examining the network gradients and comparing them to those a neutral baseline.  
#### Integrated Gradients
![Integrated Gradients](https://github.com/tal-yifat/deep-NLP-explainability/blob/main/Images/Integrated%20Gradients.png)
We can see that attributing prediction impact to specific words is helpful, but not nearly as effective as a meaningful explanation we would get from a human expert, which would refer to higher structures of meaning that specific words or tokens. For example, such an expert explanation could be something like: "injuries in which employees fall from tall structures and hurt their back tend to require longer recovery time". 
## Multi-task learning in deep neural networks
Multi-task learning (MTL) involves training models simultaneously on several related tasks, while sharing representations between the models. For instance, [Caruana (1998)](https://link.springer.com/chapter/10.1007/978-1-4615-5529-2_5) used a model that classified lane markings as an auxiliary task for the main task of learning to steer a car. MTL often improves performance and/or reduced the size of the data required to achieve a given level of performance. In some areas, the same data could be effectively used to simultaneously predict multiple targets; for example, predicting multiple economic indicators based on financial data, or the efficacy for multiple drugs based on genome data. For overviews of MTL see [here](https://arxiv.org/abs/1706.05098) and [here](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiM-q6Zx4b9AhWlVTABHRG9BbgQFnoECAcQAQ&url=https://arxiv.org/abs/2109.09138&usg=AOvVaw2e3q7TJsYwbB6nReGX37t2).   
 
To get an intuition, we can see how for humans as well, mastering one task can help us learn others that related but still different. MTL is effective because it provides implicit data augmentation, since data has different noise patterns in relation to different tasks. MTL also helps focus the model's attention on features that prove effective on multiple tasks, and therefore allowing the model to generalize better. Another mechanism that explains the effectiveness of MTL is "eavesdropping”: exploiting learned representations that are easier to learn for one task to other tasks for which they are harder to learn. Finally, MTL reduces the risk of the model overfitting to random noise. MTL can be implemented using: 
* *hard parameter sharing*, in which the hidden layers of are generally shared between all the tasks, and are followed by a small number of task-specific output layers;
* *soft parameter sharing*, in which each task has its own model and parameters, but the parameters are constrained and/or regularized to limit the distance between the parameters in different model; and
* *learned parameter sharing*, in which the network learns which of the learned parameters are shared between tasks.   

Using MTL for model explainability is a an approach that is still in its very early stages and has the promise of being much more intuitive for users than explanations based on word attribution. In this approach one or more auxiliary explanation tasks are learned in addition to the main prediction task, potentially improving the model while adding an explanation. Below is an example from [Ras et al (2021)](https://arxiv.org/abs/2004.14545).
![enter image description here](https://github.com/tal-yifat/deep-NLP-explainability/blob/main/Images/Explainability%20w%20MTL.png)
> Written with [StackEdit](https://stackedit.io/).
