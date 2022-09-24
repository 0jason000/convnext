# ConvNext

***

In January 2022, Facebook AI research and UC Berkeley jointly published an article called a convnet for the 2020s. In this article, convnext pure convolutional neural network is proposed, Compared with the current popular swing transformer, a series of experiments show that convnext achieves higher reasoning speed and better accuracy under the same flops.

The architectural definition of network refers to the following papers:

[1] Liu Z , Mao H , Wu C Y , et al. [A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545v2.pdf). 2022.

## Examples

### Train

- The following configuration for training.

```shell
mpirun -n 8 python train.py --model convnext_base --data_url /data0/imagenet2012
```

  output:

  ```text
    Epoch:[0/90], step:[2502/2502], loss:[6.946/6.946], time:1095568.310, lr:0.0049
    Epoch time:1106626.360, per step time:442.292, avg loss:6.946
    Epoch:[1/90], step:[2502/2502], loss:[6.885/6.885], time:1091818.493, lr:0.0048
    Epoch time:1091981.163, per step time:438.975, avg loss:6.885
    Epoch:[2/90], step:[2502/2502], loss:[7.026/7.026], time:1094093.746, lr:0.0047
    Epoch time:1094096.121, per step time:437.289, avg loss:7.026
    ...
  ```

### Eval

- After training, you can use test set to evaluate the performance of your model. Run eval.py to achieve this. The usage of model_type parameter is same as training process.

```text
python eval.py --model convnext_base --data_url ./imagenet2012 --ckpt_file [Your CheckPoint File Path] --num_parallel_workers 1
```

output:

```text
{'Top_1_Accuracy':0.8552283653846153,'Top_5_Accuracy':0.9777644230769231}
```
