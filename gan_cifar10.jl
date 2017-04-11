for p in ("Knet","ArgParse","Compat","GZip","Images")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet
using ArgParse
using Compat, GZip
using Images

function main(args="")

	batchsize = 100
	(xtrn,xtst,ytrn,ytst) = loaddata();

	G_net= initialize_generator_net(batchsize)
	D_net = initialize_discriminator_net(batchsize)
  info("GAN Started...")


  xtrn = traindata(defdir)
  ytrn = testdata(defdir)
	#Generate an image before any training
	#save("before_training.png",reshape(generator(G_net),(28,28)))
  x_mb = minibatch(xtrn, ytrn, o[:batchsize]);


	ep=1
	@time for i=1:length(x_mb)
		G_net = trainG(x_mb[i],D_net,G_net)
		D_net = trainD(x_mb[i],D_net,G_net)
	#	print("epoch: ",ep," loss[generative]: ",G_loss(G_net,xtrn,D_net)," loss[discriminative]: ",D_loss(D_net,xtrn,G_net)," \n");
		ep += 1
	end

	a = D_loss(D_net,xtrn,G_net)
	print("D loss after training: ",a,"\n")
	b = G_loss(G_net,xtrn,D_net)
	print("G loss after training: ",b,"\n")

	#Generate an image after training
	#save("after_training.png",reshape(generator(G_net),(28,28)))



end


function minibatch(x, y, batchsize=100; atype=Array{Float32}, xrows=784, yrows=10, xscale=255)
    xbatch(a)=convert(atype, reshape(a./xscale, xrows, div(length(a),xrows)))
    ybatch(a)=(a[a.==0]=10; convert(atype, sparse(convert(Vector{Int},a),1:length(a),one(eltype(a)),yrows,length(a))))
    xcols = div(length(x),xrows)
    xcols == length(y) || throw(DimensionMismatch())
    data = Any[]
    for i=1:batchsize:xcols-batchsize+1
        j=i+batchsize-1
        push!(data, (xbatch(x[1+(i-1)*xrows:j*xrows]), ybatch(y[i:j])))
    end
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
		for i in 1:length(D_net)
			D_net[i] -= lr * g[i]
			#axpy!(-lr,g[i],D_net[i])
		end
	end
	return D_net
end

function G_loss(G_net,x,D_net)
	G_sample = generator(G_net)
	D_fake = discriminator(G_sample,D_net)
	-mean(log(1 - D_fake))
end

lossgradient_G = grad(G_loss)

function D_loss(D_net,x,G_net)
	G_sample = generator(G_net)
	D_real  = discriminator(x,D_net)
	D_fake = discriminator(G_sample,D_net)
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
	#relu can be used?
	Z = xavier(1,100)
	G_h1 = (Z * G_net[1] .+ G_net[2]);
	G_logp = G_h1 * G_net[3] .+ G_net[4] ;
	G_prob = sigmoid(G_logp)
	#G_prob = reshape(G_prob,(size(G_prob,2),28,28))
	return G_prob
end

function initialize_generator_net()
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


function sample_noise(bs,dim;atype=Array{Float32})
  return convert(atype,randn(dimension,batchsize))
end

const defdir = joinpath(pwd(),"datasets/cifar10")

function getdata(dir)
    mkpath(dir)
    path = download("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")
    run(unpack_cmd(path,dir,".gz",".tar"))
end

function readdata(data::Vector{UInt8})
    n = Int(length(data)/3073)
    x = Array(Float64, 3072, n)
    y = Array(Int, n)
    for i = 1:n
        k = (i-1) * 3073 + 1
        y[i] = Int(data[k])
        x[:,i] = data[k+1:k+3072] / 255
    end
    x = reshape(x, 32, 32, 3, n)
    x, y
end

function traindata(dir=defdir)
    files = ["$(dir)/cifar-10-batches-bin/data_batch_$(i).bin" for i=1:5]
    all(isfile, files) || getdata(dir)
    data = UInt8[]
    for file in files
        append!(data, open(read,file))
    end
    readdata(data)
end

function testdata(dir=defdir)
    file = "$(dir)/cifar-10-batches-bin/test_batch.bin"
    isfile(file) || getdata(dir)
    readdata(open(read,file))
end


main()
