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

  
 ## Notes
 
 Although the model is minimizing the loss, the output image is not a hand-written digit image.  
 One possible cause is noted inside the loss function in the code file.  
 The main purpose of this version of the code is to set up the baseline for the GAN.  
 Therefore, the errors will be removed in the up coming versions.
 
 
