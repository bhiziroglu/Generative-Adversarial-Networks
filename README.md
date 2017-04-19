# Generative-Adversarial-Networks
Implementation of the paper "Generative Adversarial Networks, arXiv:1406.2661"  
Using Knet Library for Julia. https://github.com/denizyuret/Knet.jl  


Output (before training) | Output (after training) | Output (after training)   
------------------------ | ----------------------- | -----------------------
random weights | MNIST | CIFAR-10    
 ![Alt text](/outputs/sampleoutput.png?raw=true "Sample Output") | ![Alt text](/outputs/sampleoutput140417_3.png?raw=true "Sample Output")  ![Alt text](/outputs/sampleoutput140417_4.png?raw=true "Sample Output")  ![Alt text](/outputs/sampleoutput140417_5.png?raw=true "Sample Output") | ![Alt text](/outputs/sampleoutput140417_6.png?raw=true "Sample Output") ![Alt text](/outputs/sampleoutput140417_7.png?raw=true "Sample Output") ![Alt text](/outputs/sampleoutput140417_8.png?raw=true "Sample Output") 
[28x28 Gray Image] | [28x28 Gray Image] | [32x32 RGB Image]


```
SAMPLE RUN Thu Apr 20 02:28:11 MSK 2017 on AWS GPU g2.2xlarge

[~]$ julia gan_cifar10.jl --epochs 20
INFO: Knet using GPU 0
INFO: GAN Started...
INFO: Loading CIFAR-10...
INFO: CIFAR-10 Loaded
Initializing Discriminator weights.
Initializing Generator weights.
epoch: 1 loss[D]: 5.6787 loss[G]: 5.83609
epoch: 2 loss[D]: 5.74091 loss[G]: 5.90107
epoch: 3 loss[D]: 5.93438 loss[G]: 5.54802
epoch: 4 loss[D]: 5.95266 loss[G]: 5.49418
epoch: 5 loss[D]: 5.9691 loss[G]: 5.42383
epoch: 6 loss[D]: 5.97866 loss[G]: 5.39437
epoch: 7 loss[D]: 5.98102 loss[G]: 5.36373
epoch: 8 loss[D]: 5.98301 loss[G]: 5.3535
epoch: 9 loss[D]: 5.97895 loss[G]: 5.38156
epoch: 10 loss[D]: 5.98034 loss[G]: 5.36375
epoch: 11 loss[D]: 5.97487 loss[G]: 5.37887
epoch: 12 loss[D]: 5.96247 loss[G]: 5.43054
epoch: 13 loss[D]: 5.96378 loss[G]: 5.4194
epoch: 14 loss[D]: 5.95002 loss[G]: 5.46403
epoch: 15 loss[D]: 5.91741 loss[G]: 5.5611
epoch: 16 loss[D]: 5.89484 loss[G]: 5.62796
epoch: 17 loss[D]: 5.89259 loss[G]: 5.65407
epoch: 18 loss[D]: 5.89036 loss[G]: 5.64925
epoch: 19 loss[D]: 5.82706 loss[G]: 5.82827
epoch: 20 loss[D]: 5.80835 loss[G]: 5.89455
 91.402368 seconds (109.65 M allocations: 35.460 GB, 4.42% gc time)

```

## What Changed?
- CIFAR-10 dataset is added.
- One-hot arrays are used to figure out the number of correct guesses for networks. 
- Dropout probability is set to 0.5.
- Log probabilities are removed from the loss functions.
 
## TODO
- Output images for CIFAR-10 dataset are not as good as MNIST images. Fix that problem.
 - Losses are still not good. Try to fix that.
 - Output images are selected manually from a wide range of images created. Decide how that selection should be made.
 - Apply the model to Labeled Faces in the Wild dataset.
