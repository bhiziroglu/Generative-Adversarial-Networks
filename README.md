# Generative-Adversarial-Networks
Implementation of the paper "Generative Adversarial Networks, arXiv:1406.2661"  
Using Knet Library for Julia. https://github.com/denizyuret/Knet.jl  


Output (before training)   
-- Using MNIST dataset   
![Alt text](/outputs/sampleoutput.png?raw=true "Sample Output")    
[28x28 Image]

Output (after training)   
![Alt text](/outputs/sampleoutput140417_2.png?raw=true "Sample Output")    
[28x28 Image]    

```
SAMPLE RUN Thu Apr 14 14:56:21 MSK 2017 on AWS GPU g2.2xlarge

[ec2-user@ip-172-31-4-2 ~]$ julia gan_mnist.jl
INFO: Knet using GPU 0
INFO: GAN Started...
INFO: Loading MNIST...
Initializing Discriminator weights.
Initializing Generator weights.

epoch: 1 loss[D]: 2867.16 loss[G]: 99.8857
epoch: 2 loss[D]: 2790.19 loss[G]: 26.981
epoch: 3 loss[D]: 2774.2 loss[G]: 11.0762
epoch: 4 loss[D]: 2769.35 loss[G]: 6.25019
epoch: 5 loss[D]: 2767.25 loss[G]: 4.14593
epoch: 6 loss[D]: 2766.12 loss[G]: 3.01691
epoch: 7 loss[D]: 2765.44 loss[G]: 2.33704
epoch: 8 loss[D]: 2764.99 loss[G]: 1.88515
epoch: 9 loss[D]: 2764.67 loss[G]: 1.5665
epoch: 10 loss[D]: 2764.43 loss[G]: 1.33127
```

## What Changed?
- Activation functions for the last layer of generative model has been changed from tanh() to sigm()
- Instead of normal matrix multiplication with learning rate, update!() function is used in training.

 ## TODO
 - The discriminative loss is not descending, fix that problem
 - The generative loss should follow discriminative loss according to the paper. Find out how one is decreasing and the other one is stable.
 - Apply the model to other two datasets.
