# Generative-Adversarial-Networks
Implementation of the paper [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661 "arXiv")
Using [Knet](https://github.com/denizyuret/Knet.jl "Knet Github Repo") Library for Julia.   

## Introduction

- GANs are used to generate realistic looking samples.    
- MNIST model uses MLP to generate samples.   
- CNN is used for other datasets.   
- The model must be trained on a GPU machine.
- If the dataset does not exist in the current directory, it will be downloaded.

## Usage

```
$ julia gan_mnist.jl

$ julia gan_faces.jl

$ julia gan_cifar.jl
```

NOTE: To run the code, [this line](https://github.com/denizyuret/Knet.jl/blob/master/src/conv.jl#L355) should be replaced with   `size(w,N-1)` on your current Knet installation.

## Generated Samples
![Alt text](/outputs/mnist_sample1.png?raw=true "Sample Output")   


![Alt text](/outputs/mnist_sample2.png?raw=true "Sample Output")      


![Alt text](/outputs/mnist_sample3.png?raw=true "Sample Output")      


![Alt text](/outputs/cifar_sample1.png?raw=true "Sample Output")   

![Alt text](/outputs/cifar_sample2.png?raw=true "Sample Output")    


 
## 📝 TODO
- Output images for CIFAR-10 dataset have low resolution. 
- The model for Labeled Faces in The Wild dataset only works if the dataset is in the current directory. Will fix that.


## 📚 Tutorial
- A tutorial for Generate Adversarial Networks can be found [here](https://arxiv.org/abs/1701.00160 "arXiv").

## Related Works
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434 "arXiv")
