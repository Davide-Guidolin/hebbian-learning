# Neuron-Centric Hebbian Learning for Convolutional Networks

This repository contains the work for my Master Thesis, entitled _Neuron-Centric Hebbian Learning for Convolutional Networks_. 

## Abstract
Convolutional neural networks (CNNs) have revolutionized computer vision tasks, achieving remarkable performance. However, their training algorithm, backpropagation, faces limitations regarding biological plausibility. Unlike the localized learning observed in the brain, backpropagation relies on
global error signals, hindering its biological relevance. Hebbian learning emerges as a promising alternative, inspired by the principle of ”fire together, wire together.” It updates connections based solely on the activities of presynaptic and postsynaptic neurons, offering a more biologically plausible approach. However, integrating Hebbian learning with CNNs presents a significant challenge. The core operation of CNNs, convolution, involves multiple input neurons influencing a single output, creating ambiguity in defining presynaptic and postsynaptic activities within the Hebbian framework. This
work tackles this challenge by proposing a novel approach to explore Hebbian learning in CNNs. We use the work by Cunegatti et al. to unroll convolutional layers into equivalent linear layers, enabling a clear definition of presynaptic and postsynaptic activities. We then employ a variation of HebbianABCD, a recent advancement in Hebbian learning algorithms, to train the unrolled network on a reinforcement learning task. Our custom HebbianABCD algorithm reduces the number of parameters with respect to the standard HebbianABCD and is designed to keep the shared weights property of convolutional layers. Additionally, we leverage Evolution Strategy as an optimization algorithm to fine-tune the Hebbian parameters. This choice reinforces the biologically inspired nature of our approach, aiming to achieve learning dynamics that closely resemble those observed in nature. We also train the unrolled network on two classification tasks to test a weight sharing strategy similar to the one used for reinforcement learning but without the need for the ABCD parameters of HebbianABCD.
Our work is structured in the following way. We start by introducing the limitations of backpropagation and examining how these limitations can be addressed through the application of Hebbian learning. In particular, we’ll focus on the difficulties involved in integrating Hebbian learning with
convolutional networks. We provide an overview on Hebbian Learning and all its variants namely anti-Hebbian, Hebbian winner-takes-all, Hebbian principal component analysis and HebbianABCD.
We then explore other works that use Hebbian learning to train convolutional networks (chapter 2). In chapter 3 we describe the two image classification tasks, on the CIFAR10 and CIFAR100 datasets, and the reinforcement learning task, on the OpenAI gym CarRacing environment, used to measure the performances of our approach. We also describe in details the methods and the architectures used as baselines. Then, in chapter 4, we describe the method proposed by Cunegatti et al. to create an unrolled input-aware bipartite graph encoding for each layer of a neural network and we show how we use this encoding to create linear layers from convolutional ones. We also highlight the differences between the original networks and the unrolled ones with a focus on the number of parameters before and after the unrolling. In chapter 5 we dive into HebbianABCD and its variants, then we propose a new neuron-centric variant that drastically reduces the number of Hebbian parameters. We also explore a method to update the network’s weights using HebbianABCD while keeping the shared weights property of convolutional layers. We will also show part of the HebbianABCD implementation where we worked to improve the execution time. Then we will explain how evolution strategy works, focusing on the implementation that we chose for our experiments. Additionally, we explore another method
to perform weight sharing without using HebbianABCD, and we experiment it using Soft Hebbian Winner-Takes-All as learning rule. In chapter 6 we briefly describe the experimental setups and we report the results obtained using the baselines and our method. We discuss why our approach is not as competitive as the baseline methods and propose hypotheses to explain this. Finally in chapter 7 we summarize our work and we conclude with the pros and cons of our approach along with some ideas for future directions to further explore and potentially address the limitations identified.

## Experiments

To run the experiments move into the `graph_encoding_abcd/` folder.

### Training on CIFAR
To train the network on a CIFAR dataset using soft Hebbian WTA, sharing the weights with the `max` aggregation function use the following command:

```bash
python3 run_exp.py --dataset CIFAR10 --epochs 51 --device cuda --aggregation_function max --bp_last_layer --softhebb --bp_lr 0.001 --softhebb_lr 0.01
```
where

* `--dataset` can be `CIFAR10` or `CIFAR100`
* `--epochs` is `51` i.e. `1` for soft WTA and `50` for backpropagation. If you want to change the number of soft WTA epochs you need to manually change the file `softhebb_train.py` at line `154`, by setting the value of `only_bp = True` when you want to train the last layer with backpropagation.
* `--aggregation_function` is the aggregation function used for weight sharing. It can be `max`, `min` and `median`.
* `--bp_last_layer` is needed to train the last layer with backpropagation.
* `--softhebb` is needed to use soft WTA.
* `--bp_lr` is the learning rate using during the training of backpropagation, halved different times during the training. To change the halving percentages, change the `lr_halve` variable at line `151` in the file `softhebb_train.py`.
* `--softhebb_lr` is the learning rate used during the soft WTA updates.

### Training on CarRacing
To train the network on CarRacing-v2 using our neuron-centric HebbianABCD learning rule, optimized with Evolution Strategy, use the following command:

```bash
python3 run_exp.py --dataset CarRacing --population_size 300 --num_threads 4 --epochs 200 --device cuda --saving_path params/cifar/min/ --aggregation_function min
```

where 

* `--population_size` is the number of individuals used for the evolution strategy.
* `--num_threads` is the number of parallel evaluations to run. You need to make sure that all the `num_threads` networks fit in memory.
* `--epochs` is the number of iterations to run.
* `--device` can be `cuda` if using the GPU or `cpu`.
* `--saving_path` is the path where the parameters will be saved.
* `--aggregation_function` is the aggregation function used for weight sharing. It can be `max`, `min` and `median`.

Other parameters are:

* `--sigma` is the standard deviation of the noise added during the mutation in the ES.
* `--abcd_learning_rate` is the learning rate used to update the ABCD parameters.
* `--abcd_lr_decay` is the decay factor for `abcd_learning_rate`.


If, for some reason, you need to stop the training and resume it you can run the same command with the `--resume_file CHECKPOINT_PATH` parameter, with `CHECKPOINT_PATH` the path to the checkpoint from which the training will be resumed. All the other parameters are taken from the checkpoint file.