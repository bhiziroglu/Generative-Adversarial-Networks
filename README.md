# Generative-Adversarial-Networks
Implementation of the paper "Generative Adversarial Networks, arXiv:1406.2661"  
Using Knet Library for Julia. https://github.com/denizyuret/Knet.jl  


Output (before training)   
-- Using MNIST dataset   
![Alt text](/outputs/sampleoutput.png?raw=true "Sample Output")    
[28x28 Image]

Output (after training)   
![Alt text](/outputs/sampleoutput140417_3.png?raw=true "Sample Output")    
[28x28 Image]



```
SAMPLE RUN Tue Apr 18 21:50:33 MSK 2017 on AWS GPU g2.2xlarge

[~]$ julia gan_mnist.jl --epochs 15
INFO: Knet using GPU 0
INFO: GAN Started...
INFO: Loading MNIST...
Initializing Discriminator weights.
Initializing Generator weights.
epoch: 1 loss[D]: 42.4067 loss[G]: 10.597
epoch: 2 loss[D]: 42.3867 loss[G]: 10.5967
epoch: 3 loss[D]: 42.3867 loss[G]: 10.5967
epoch: 4 loss[D]: 42.3867 loss[G]: 10.5967
epoch: 5 loss[D]: 42.3867 loss[G]: 10.5967
epoch: 6 loss[D]: 42.3867 loss[G]: 10.5967
epoch: 7 loss[D]: 42.3867 loss[G]: 10.5967
epoch: 8 loss[D]: 42.3867 loss[G]: 10.5967
epoch: 9 loss[D]: 42.3867 loss[G]: 10.5967
epoch: 10 loss[D]: 42.3867 loss[G]: 10.5967
epoch: 11 loss[D]: 42.3867 loss[G]: 10.5967
epoch: 12 loss[D]: 42.3869 loss[G]: 10.5967
epoch: 13 loss[D]: 42.3867 loss[G]: 10.5967
epoch: 14 loss[D]: 42.3867 loss[G]: 10.5967
epoch: 15 loss[D]: 42.3867 loss[G]: 10.5967
136.510100 seconds (159.31 M allocations: 50.402 GB, 3.39% gc time)

```

## What Changed?
- User defined learning rate is not used anymore. Instead, Adam() is called without any input.
- Dropout added.

 ## TODO
 - Losses are stable. Fix that problem
 - Output images are selected manually from a wide range of images created. Decide how that selection should be made.
 - Apply the model to other datasets.
