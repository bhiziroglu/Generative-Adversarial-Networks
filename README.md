# Generative-Adversarial-Networks

# 📓 Introduction
Implementation of the paper "Generative Adversarial Networks, arXiv:1406.2661"  
Using Knet Library for Julia. https://github.com/denizyuret/Knet.jl  


Output (before training) | Output (after training) | Output (after training)   
------------------------ | ----------------------- | -----------------------
random weights | MNIST | CIFAR-10    
 ![Alt text](/outputs/sampleoutput.png?raw=true "Sample Output") | ![Alt text](/outputs/sampleoutput280417_1.png?raw=true "Sample Output")  ![Alt text](/outputs/sampleoutput280417_2.png?raw=true "Sample Output")  ![Alt text](/outputs/sampleoutput280417_3.png?raw=true "Sample Output") | ![Alt text](/outputs/sampleoutput280417_4.png?raw=true "Sample Output") ![Alt text](/outputs/sampleoutput280417_5.png?raw=true "Sample Output") ![Alt text](/outputs/sampleoutput280417_6.png?raw=true "Sample Output") 
[28x28 Gray Image] | [28x28 Gray Image] | [32x32 RGB Image]


```
SAMPLE RUN Thu Apr 28 03:38:42 MSK 2017 on AWS GPU g2.2xlarge

[~]$ julia gan_mnist.jl -- epoch 50
INFO: Knet using GPU 0
INFO: GAN Started...
Dict{Symbol,Any}(Pair{Symbol,Any}(:print,true),Pair{Symbol,Any}(:batchsize,100),Pair{Symbol,Any}(:gencnt,100),Pair{Symbol,Any}(:epochs,100),Pair{Symbol,Any}(:lr,0.1),Pair{Symbol,Any}(:atype,"KnetArray{Float32}"))
INFO: Loading MNIST...
Initializing Discriminator weights.
Initializing Generator weights.
epoch: 1 loss[D]: 0.129049 loss[G]: 4.12751
epoch: 2 loss[D]: 0.111942 loss[G]: 4.95543
epoch: 3 loss[D]: 0.186384 loss[G]: 4.63559
epoch: 4 loss[D]: 0.465959 loss[G]: 2.06095
epoch: 5 loss[D]: 0.594776 loss[G]: 1.12168
…….
epoch: 46 loss[D]: 0.524244 loss[G]: 1.33503
epoch: 47 loss[D]: 0.52226 loss[G]: 1.34622
epoch: 48 loss[D]: 0.519366 loss[G]: 1.34495
epoch: 49 loss[D]: 0.515052 loss[G]: 1.37154
epoch: 50 loss[D]: 0.515786 loss[G]: 1.36772
 991.402368 seconds (809.65 M allocations: 98.460 GB, 4.52% gc time)

```
![Alt text](/outputs/sampleoutput280417_matlab.png?raw=true "Sample Output")


## What Changed?
- Labels are no longer used.
- tanh() as an activation function is added to every layer of generator.
- Input images are normalized between [-1,1]   
- Log probabilities are added to the loss functions. 

## ☑ TODO
- [X] Losses are logical and agree with the paper.
- [X] Output images are selected manually from a wide range of images created. Decide how that selection should be made.
- [ ] MLPs take very long time to produce good output for CIFAR-10 images. Try to use CNNs.   
- [ ] Apply the model to Labeled Faces in the Wild dataset.
