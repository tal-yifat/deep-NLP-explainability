# Using Multi-task learning to explain deep NLP models
## Overview 
### The research problem
Explainability has been a big research topic in recent years and saw the development several approaches to explaining deep learning models. While such methods have helped researchers gain insight into deep NLP models, these methods tend to be inaccessible to business users. The purpose of this research project is using [multi-task-learning](https://arxiv.org/abs/1706.05098) to develop explainability methods that would be accessible to and incorporate the input of business users.
### The data
The research uses [a public dataset on mine accidents from the U.S. Mine Safety and Health Administration](https://www.msha.gov/data-and-reports/mine-data-retrieval-system). The data includes a text narrative of the accident and structured data on the accident and its outcomes. The main task of the developed models is to predict, based on the narrative alone, whether a worker will lose at least 90 days of work as a result of the accident. 
### The business problem
The research project targets hypothetical business users who process disability insurance claims. For disability insurers, a large proportion of the cost comes from a relatively small number of claims in which workers take a long time to return to work. Identifying early on cases that have a high risk of turning into long-term disability claims could allow early intervention that would help customers recover more quickly and save money to the company. The project tests the hypothesis that textual data could help with early detection of such cases.  


> Written with [StackEdit](https://stackedit.io/).
