# Class-incremental Learning with Pre-allocated Fixed Classifiers

This repo contains the code of "Class-incremental Learning with Pre-allocated Fixed Classifiers" (ICPR2020).

Our code relies on the framework of "Re-evaluating Continual Learning Scenarios: A Categorization and Case for Strong Baselines" paper available at https://github.com/GT-RIPL/Continual-Learning-Benchmark

Refer to our paper for more details: https://arxiv.org/abs/2010.08657

If you have any question, feel free to open an issue.

<p align="center">
  <img src="https://github.com/DigiTurk84/class-incremental-polytope/blob/main/img/intro_top.png?raw=true">
</p>

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

1. clone the repository available at: https://github.com/GT-RIPL/Continual-Learning-Benchmark (at the time of writing we tested against commit d78b997)
```
git clone https://github.com/GT-RIPL/Continual-Learning-Benchmark
cd Continual-Learning-Benchmark
git checkout d78b997
```

2. merge the files available in this repo with the code cloned at step 1

## Usage

follow instruction at https://github.com/GT-RIPL/Continual-Learning-Benchmark#usage and use our modules with the following arguments.

### Model

To use one of the models with fixed classifier, add these options

```
--model_type fixed_model
--model_name model_name
```
where fixed model is one of the following:
1. fixed_resnet
2. fixed_mlp
3. fixed_lenet

and model_name is one of the classes names defined in the respective files

### Agent

to use our naive reharsal with simplex, add these options
```
--agent_type fixed_agent
--agent_name fixed_agent_class 
```
where fixed_agent_class is one of the following:

1. Fixed_Simplex_Naive_Rehearsal_100
2. Fixed_Simplex_Naive_Rehearsal_1100
3. Fixed_Simplex_Naive_Rehearsal_1400
4. Fixed_Simplex_Naive_Rehearsal_5600

## Authors

- Federico Pernici <federico.pernici at unifi.it> [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/FedPernici.svg?style=social&label=Follow%20%40FedPernici)](https://twitter.com/FedPernici)

- Matteo Bruni <matteo.bruni at unifi.it>
- Claudio Baecchi <claudio.baecchi at unifi.it>
- Francesco Turchini <francesco.turchini at unifi.it>
- Alberto Del Bimbo <alberto.delbimbo at unifi.it>


## Citing

Please kindly cite our paper if this repository is helpful.
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
