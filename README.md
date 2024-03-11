# Neuron-Centric Hebbian Learning for Convolutional Networks

To run the experiments move into the `graph_encoding_abcd/` folder.

## Training on CIFAR
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

## Training on CarRacing
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


If for some reason you need to stop the training and resume it you can run the same command with the `--resume_file CHECKPOINT_PATH` parameter, with `CHECKPOINT_PATH` the path to the checkpoint from which the training will be resumed. All the other parameters are taken from the checkpoint file.