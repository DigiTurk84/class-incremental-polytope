# Class-incremental Learning with Pre-allocated Fixed Classifiers

This repo contains the original code used for the experiments of "Class-incremental Learning with Pre-allocated Fixed Classifiers".

Our code relies on the framework of "Re-evaluating Continual Learning Scenarios: A Categorization and Case for Strong Baselines" paper available at https://github.com/GT-RIPL/Continual-Learning-Benchmark

Refer to our paper for more details: <link to paper will be updated as soon the conference will publish it online>
If you have any question, feel free to open an issue.

## Abstract
> In class-incremental learning, a learning agent faces a stream of data with the goal of learning new classes while not forgetting  previous ones. 
Neural networks are known to suffer under this setting, as they forget previously acquired knowledge. 
To address this problem, effective methods exploit past data stored in an episodic memory while expanding the final 
classifier nodes to accommodate the new classes. In this work, we substitute the expanding classifier with a novel fixed classifier
in which a number of pre-allocated output nodes are subject to the classification loss right from the beginning of the learning phase. 
Contrarily to the standard expanding classifier, this allows: (a) the output nodes of future unseen classes to firstly see negative 
samples since the beginning of learning together with the positive samples that incrementally arrive; 
(b) to learn features that do not change their geometric configuration as novel classes are incorporated in the learning model.
Experiments with public datasets show that the proposed approach is as effective as the expanding classifier while
exhibiting novel intriguing properties of the internal feature representation that are otherwise not-existent. 
Our ablation study on pre-allocating a large number of classes further validates the approach.


## Preparation

1. clone the repository available at: https://github.com/GT-RIPL/Continual-Learning-Benchmark at commit d78b997
2. merge the files available in this repo with the code cloned at step 1

## Usage

follow instruction at https://github.com/GT-RIPL/Continual-Learning-Benchmark#usage
to use our models/agent add these at the command line:

```
```



## Citing

Please kindly cite our paper if this paper is helpful.
```
@article{pernici2020icpr,
  author    = {Federico Pernici and
               Matteo Bruni and
               Claudio Baecchi and
               Francesco Turchini and
               Alberto Del Bimbo},
  title     = {Class-incremental Learning with Pre-allocated Fixed Classifiers},
  booktitle = {25th International Conference on Pattern Recognition, {ICPR} 2020,
               Milan, Italy, January 10-15, 2021},
  publisher = {{IEEE} Computer Society},
  year      = {2020},
}
```
