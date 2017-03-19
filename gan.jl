for p in ("Knet","ArgParse","Compat","GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

#Package MLDatasets for Julia
#Makes it easier to access popular datasets
#Pkg.clone("https://github.com/JuliaML/MLDatasets.jl.git")

using Knet
using ArgParse
using Compat, GZip
#using Plots
using MLDatasets

function main(args="")
	
	batchsize = 1
	#getting the data using MLDataset
	xtrn , ytrn = MNIST.traindata()
	
	G_net= initialize_generator_net(batchsize)
	D_net = initialize_discriminator_net(batchsize)
#	print("SIZE OF D_net ==> ",size(D_net),"\n")
	
	x_mb = minibatch(xtrn[:,:,1:20000]) #batchsize = 100 [default]
	
	
	
	
	a = D_loss(D_net,xtrn,G_net)
	print("D loss prior: ",a,"\n")
	b = G_loss(G_net,xtrn,D_net)
	print("G loss prior: ",b,"\n")
	
	
	first_img = generator(G_net)
	print("noise image\n")
	print(first_img)
	
	#c = lossgradient_D(D_net,xtrn,G_net)
	#print("loss grad size : ",size(c[2]),"\n")
	#print("D_net : ",size(D_net[2]),"\n")
	
	@time for i=1:length(x_mb)
		G_net = trainG(x_mb[i],D_net,G_net)
		D_net = trainD(x_mb[i],D_net,G_net)
	end
	
	a = D_loss(D_net,xtrn,G_net)
	print("D loss post: ",a,"\n")
	b = G_loss(G_net,xtrn,D_net)
	print("G loss post: ",b,"\n")
	
	last_img = generator(G_net)
	print("last image\n")
	print(last_img)
	
	
	
end

function minibatch(X,bs=100)
	#takes raw input (X) and gold labels (Y)
	#returns list of minibatches (x, y)
	data = Any[]
	#start of step 1
	for i=1:bs:size(X, 3)
		bl = i + bs - 1 <= size(X, 3) ? i + bs - 1 : size(X, 3)
		push!(data, (X[:,:, i:bl]))
	end
	#end of step 1
	return data
end


function trainG(xtrn,D_net,G_net,lr=-.2,epochs=1)
	for epoch=1:epochs
		g = lossgradient_G(G_net,xtrn,D_net)
		for i in 1:length(G_net)
			axpy!(-lr,g[i],G_net[i])
		end
	end
	return G_net
end



function trainD(xtrn,D_net,G_net,lr=.1 , epochs=5)
	for epoch=1:epochs
		g = lossgradient_D(D_net,xtrn,G_net)
#		print("current g=>",g,"\n")
#		print("epoch(",i,") loss: ",D_loss(xtrn[:,:,128],D_net,G_net)," grad: ",g,"\n")
		for i in 1:length(D_net)
			axpy!(-lr,g[i],D_net[i])
		end
	end
	return D_net
end

function G_loss(G_net,x,D_net)
	G_sample = generator(G_net)
	D_fake = discriminator(G_sample,D_net)
	-mean(log(D_fake))
end

lossgradient_G = grad(G_loss)

function D_loss(D_net,x,G_net)
	G_sample = generator(G_net)
	D_real  = discriminator(x,D_net)
	D_fake = discriminator(G_sample,D_net)
	#there is a problem here
	#fake image is just a single image whereas the real image is a
	#batch of images
	#TODO: solve this problem
	-mean(log(D_real) .+ log(1 - D_fake))
end

lossgradient_D = grad(D_loss)

#returns a probability which tells whether the input image	
#is from the real dataset or a generated one
function discriminator(x,D_net)
	#reshape the input image and take its transpose
	#so that it can be multiplied with the first layer of the Dnet
	x = reshape(x,(784,size(x,3)))
	x = x' #transpose
	#now x is in the form (bs,784)
	D_h1 = max(0, (x * D_net[1] .+ D_net[2] ) );
	D_logit = D_h1 * D_net[3] .+ D_net[4];
	D_prob = sigmoid(D_logit)
	return D_prob
end

#it takes N-dimensional vector where N is an arbitrary number
#return a 784 dimensional MNIST image
function generator(G_net)
	#100 can be changed here
	Z = xavier(1,100)
	G_h1 = (Z * G_net[1] .+ G_net[2]);
	G_logp = G_h1 * G_net[3] .+ G_net[4] ;
	G_prob = sigmoid(G_logp)
	#G_prob = reshape(G_prob,(size(G_prob,2),28,28))
	return G_prob	
end

function initialize_generator_net(bs)
	G = Any[]
	#100 here can be changed
	G1 = xavier(100,bs) #G_W1
	G2 = zeros(1,bs) #G_B1
 	G3 = xavier(bs,784) #G_W2
	G4 = zeros(1,784) #G_B2
	push!(G,G1)
	push!(G,G2)
	push!(G,G3)
	push!(G,G4)
	#map(a->convert(KnetArray{Float32},a), D)
	return G
end

function xavier(a...)
    w = rand(a...)
     # The old implementation was not right for fully connected layers:
     # (fanin = length(y) / (size(y)[end]); scale = sqrt(3 / fanin); axpb!(rand!(y); a=2*scale, b=-scale)) :
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in xavier")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w)) # Caffe disagrees: http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html#details
        fanin = div(length(w), fanout)
    end
    # See: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    s = sqrt(2 / (fanin + fanout))
    w = 2s*w-s
end

function initialize_discriminator_net(bs)
	D = Any[]
	D1 = xavier(784,bs) #D_W1
	D2 = zeros(1,bs) #D_B1
	D3 = xavier(bs,1) #D_W2
	D4 = zeros(1,1) #D_B2
	push!(D,D1)
	push!(D,D2)
	push!(D,D3)
	push!(D,D4)
	#map(a->convert(KnetArray{Float32},a), G)
	return D #returns the generator net
end



function sigmoid(z)
  return 1.0 ./ (1.0 .+ exp(-z))
end

main()


