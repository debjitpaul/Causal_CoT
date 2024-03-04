# Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning :rocket: 

## Repo in Progress ...

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![MIT License](https://img.shields.io/github/license/m43/focal-loss-against-heuristics)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2402.13950-b31b1b.svg)](https://arxiv.org/abs/2402.13950)


Official implementation of 📖 [Making Reasoning Matter:
Measuring and Improving Faithfulness of Chain-of-Thought Reasoning](https://arxiv.org/pdf/2304.01904.pdf) 

🔗 [Blog Post](https://debjitpaul.github.io/reasoningmatter)


## 🔍 Contents

- [🌟 Overview](#overview)
- [🌟 Method](#method)
- [🔥 Dependencies](#dependencies)
- [🔥 Setup](#setup)
- [🔥 Data](#data)
- [🔥 Models](#models)
- [🚩 Citation ](#citation)



## Dependencies

- compatible with python 3.8
- dependencies can be installed using `requirements.txt`
- The codebase is built around [Hugging Face](https://huggingface.co/) ecosystem and [wandb](https://wandb.ai/site) (for monitoring and experiment management).

Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv -p python3 venv
$ source venv/bin/activate
```

Install all the required packages:

```shell
$ pip install -r requirements.txt
```

## Data 

| Data                       | Reference                                                    | Output  | Description                                                  |
| :-------------------------- | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| GSM8K          | [📖](https://arxiv.org/pdf/2103.07191.pdf) , [🗂️](https://github.com/arkilpatel/SVAMP/tree/main/data/mawps-asdiv-a_svamp_without_questions), [🔗](https://github.com/arkilpatel/SVAMP) | CoT (z) and Answers (y) | Generate an equation given a math word problem question |
| StrategyQA          | [📖](https://crfm-helm.readthedocs.io/en/latest/) , [🗂️](https://github.com/stanford-crfm/helm), [🔗](https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/scenarios) | Reasoning steps (z) and Conclusion (y) | This task requires the model to perform deductive reasoning and generate intermediate reasoning steps z and conclusions y using closed-world rules and facts. |
| Causal Understanding          | [📖](https://aclanthology.org/2021.emnlp-main.54.pdf) , [🗂️](https://tinyurl.com/moral-stories-data), [🔗](https://huggingface.co/datasets/demelin/moral_stories) | Moral Norm (z) and Moral Action (y) | Given a context x consisting of a situation, an intention, and an immoral action, the model needs to generate the moral norm z and the moral action y |


## Setup


Start by cloning the repository:

```bash
git clone git@github.com:debjitpaul/Causal_CoT.git
```



  ## Citation

  ```
  @misc{debjit2024frodo,
    title={Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning}, 
    author={Debjit Paul, Robert West, Antoine Bosselut and Boi Faltings}
    year={2024},
    eprint={2402.13950},
    archivePrefix={arXiv},
    primaryClass={cs.CL}}
  ```
