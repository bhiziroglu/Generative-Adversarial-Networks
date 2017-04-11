# Generative-Adversarial-Networks
Implementation of the paper "Generative Adversarial Networks, arXiv:1406.2661"  
Using Knet Library for Julia. https://github.com/denizyuret/Knet.jl  


Output (before training)   
-- Using MNIST dataset   
![Alt text](/outputs/sampleoutput.png?raw=true "Sample Output")    
[28x28 Image]

Output (after training)   
![Alt text](/outputs/sampleoutputtrained.png?raw=true "Sample Output")    
[28x28 Image]    

```
SAMPLE RUN Tue Mar 21 22:29:30 MSK 2017   

INFO: Knet using GPU 0  
INFO: Loading MNIST...  
D loss prior: 1.3783718453599483  
G loss prior: 0.6758804389903623  
  
epoch: 1 loss[generative]: 0.22450022063780317 loss[discriminative]: 0.9754884350477281  
epoch: 2 loss[generative]: 0.04377112889020878 loss[discriminative]: 0.6507480850680244  
epoch: 3 loss[generative]: 0.023792791168789174 loss[discriminative]: 0.527970792972902   
epoch: 4 loss[generative]: 0.012100225170517287 loss[discriminative]: 0.45077804478487987  
.  
.  
epoch: 10 loss[generative]: 0.0033129904546181374 loss[discriminative]: 0.22686330041370548    
.  
.  
epoch: 100 loss[generative]: 0.00019555105638620874 loss[discriminative]: 0.021619116341306782  
```

## What Changed?
- Created baseline for the two other datasets: CIFAR-10 and Labeled Faces In The Wild.   
- Removed sigmoid() function explicitly created since its already present in Knet.   
- Used xavier() for weight initialization and randn() for sampling noise Z.
  
 ## TODO
 - Change loss functions since they do not make the model create acceptable outputs.   
 - The loss function is exactly the same as stated in the paper but I will try to concatenate the loss probabilities of real and fake data together. Otherwise, the probabilities are seperated from eachother which makes the discriminator act as two different models whereas it is a single model trying to figure out whether a presented data is fake or real.   
 - Change the variable names so the implementation looks better and easier to understand.   
 - Get rid of the unnecessary comments in the master branch.
 
