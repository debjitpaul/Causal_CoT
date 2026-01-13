# Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning :rocket: 

## Repo in Progress ...

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![MIT License](https://img.shields.io/github/license/m43/focal-loss-against-heuristics)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2402.13950-b31b1b.svg)](https://arxiv.org/abs/2402.13950)


Official implementation of 📖 [Making Reasoning Matter:
Measuring and Improving Faithfulness of Chain-of-Thought Reasoning](https://arxiv.org/pdf/2304.01904.pdf) 

🔗 [Blog Post](https://debjitpaul.github.io/reasoningmatter)

![Image](img/image.png)


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

| Data                       | Reference                                                    | 
| :-------------------------- | ------------------------------------------------------------ |
| GSM8K          | [📖](https://arxiv.org/pdf/2110.14168.pdf) , [🗂️](https://github.com/openai/grade-school-math/tree/master/grade_school_math/data)| 
| StrategyQA          | [📖]() , [🗂️](https://gith), [🔗](https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/scenarios) | 
| Causal Understanding          | [📖](), [🗂️](https://tinyurl.com/moral-stories-data), [🔗](https://huggingface.co/datasets/demelin/moral_stories) | 
| OpenBookQA          | [📖](), [🗂️](https://tinyurl.com/moral-stories-data), [🔗](https://huggingface.co/datasets/demelin/moral_stories) |
| QuaRel          | [📖](), [🗂️](https://tinyurl.com/moral-stories-data), [🔗](https://huggingface.co/datasets/demelin/moral_stories) | 
| QASC         | [📖](), [🗂️](https://tinyurl.com/moral-stories-data), [🔗](https://huggingface.co/datasets/demelin/moral_stories) | 

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
