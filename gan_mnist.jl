for p in ("Knet","ArgParse","Compat","GZip","Images")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet
using ArgParse
using Compat, GZip
using Images


function main(args)
    s = ArgParseSettings()
    s.description = "Implementation of the paper Generative Adversarial Networks [https://arxiv.org/abs/1406.2661] \nUsing Knet Library in Julia";
    s.exc_handler=ArgParse.debug_handler

    @add_arg_table s begin
      ("--epochs"; arg_type=Int; default=10)
      ("--batchsize"; arg_type=Int; default=100)
      ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
      ("--lr"; arg_type=Float32; default=Float32(0.01))
      ("--gencnt"; arg_type=Int; default=2; help="Number of images that generator function creates.")
      ("--print"; default=true ; help="Set false to turn off creating output images")
    end

  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  atype = eval(parse(o[:atype]))
  info("GAN Started...")

  (xtrn,xtst,ytrn,ytst)=loaddata()
  trn = minibatch(xtrn, ytrn, o[:batchsize];atype=atype)
  tst = minibatch(xtst, ytst, o[:batchsize];atype=atype)

  sizeX = size(trn[1][1],1)
  sizeD = 128 #user dependent
  sizeG = 128 #user dependent
  sizeZ = 100

  Dnet = initalize_weights_D(sizeX,sizeD,atype);
  Gnet = initalize_weights_G(sizeX,sizeZ,sizeG,atype);


  Z = samplenoise(size(trn[1][1])[2],sizeZ,atype,winit=0.01)

  #ADAM
  Dnetopt = map(x->Adam(;lr=o[:lr]), Dnet)
  Gnetopt = map(x->Adam(;lr=o[:lr]), Gnet)

  #generate a sample

for epoch=1:o[:epochs]
  shuffle!(trn)
  lossD = 0
  lossG = 0
  for i = 1:length(trn)
    x = convert(atype,trn[i][1])
    lossD += D_loss(Dnet,sizeZ,x,o,Gnet,atype)
    lossG += G_loss(Gnet,Dnet,sizeZ,atype,o)
    trainD(Dnet,Gnet,x,sizeZ,Dnetopt,o,atype)
    trainG(Gnet,Dnet,sizeZ,Gnetopt,o,atype)
  end
  @printf("epoch: %d loss[D]: %g loss[G]: %g\n",epoch,lossD,lossG)

end

end


function trainG(Gnet,Dnet,sizeZ,Gnetopt,o,atype)
  g = G_lossgradient(Gnet,Dnet,sizeZ,atype,o)
  	for i in 1:length(Gnet)
			Gnet[i] -= o[:lr] * g[i]
			#axpy!(-lr,g[i],D_net[i])
		end
  #for i=1:length(Gnet)
  #  update!(Gnet[i],g[i],Gnetopt[i])
  #end
end


#train Discriminative net only once
function trainD(Dnet,Gnet,x,sizeZ,Dnetopt,o,atype)
  g = D_lossgradient(Dnet,sizeZ,x,o,Gnet,atype)
  #for i=1:length(Dnet)
  #  update!(Dnet[i],g[i],Dnetopt[i])
  #end
  for i in 1:length(Dnet)
    Dnet[i] -= o[:lr] * g[i]
    #axpy!(-lr,g[i],D_net[i])
  end
end

function print_output(epoch,Gnet,sizeZ,atype,o)
  gg = generator(Gnet,sizeZ,o[:batchsize],atype,o[:gencnt])
  gg = (gg+1)/2;
  gg = min(1,max(0,gg))
  gg = convert(Array{Float64},gg)
  gg = gg[:,1:1]
  gg = reshape(gg,(28,28))
  save(@sprintf("output%d.png",epoch),gg)

end

#discriminator loss
function D_loss(Dnet,sizeZ,x,o,Gnet,atype)
	G_sample = generator(Gnet,sizeZ,o[:batchsize],atype,o[:gencnt])
	D_real  = log(discriminator(Dnet,x,atype))
	D_fake = log(1-discriminator(Dnet,G_sample,atype))
  #D_real = sum(D_real)/size(D_real,2)
  #D_fake = sum(D_fake)/size(D_fake,2)
  #=println("size of d real")
  println(size(D_real))
  println("size of d fake")
  println(size(D_fake))
  =#
  -(sum(D_real)/size(D_real,2))-(sum(D_fake)/size(D_fake,2))
end

D_lossgradient = grad(D_loss)

#generator loss
function G_loss(Gnet,Dnet,sizeZ,atype,o)
	G_sample = generator(Gnet,sizeZ,o[:batchsize],atype,o[:gencnt])
	D_fake = discriminator(Dnet,G_sample,atype) #fake prob
	-sum(log(1-D_fake))/size(D_fake,2) #min(log(1-D_fake))
end

G_lossgradient = grad(G_loss)

#returns a probability which tells whether the input image
#is from the real dataset or a generated one
function discriminator(Dnet,x,atype)
  #reshape has been removed
  D_h1 = tanh(Dnet[1] * x .+ Dnet[2] );
	D_logit = Dnet[3] * D_h1 .+ Dnet[4];
	D_prob = D_logit
  return sigm(D_prob)/size(x,2)
end


#it takes N-dimensional vector where N is an arbitrary number
#N is the dimension of the Z vector
#return a 784 dimensional MNIST image
function generator(Gnet,sizeZ,bs,atype,gencnt)
  Z = samplenoise(gencnt,sizeZ,atype) #gencnt is used defined
  #  println("size Z in generator")
  #  println(size(Z))
  #desired number of output images
  #100 can be changed here
  #relu can be used?
  #  println("size of G")
  #  for i=1:length(Gnet)
  #    println(size(Gnet[i]))
  #  end
  G_h1 = tanh(Gnet[1] * Z .+ Gnet[2]);
	G_logp = Gnet[3] * G_h1 .+ Gnet[4] ;
  #changed sigmoid to tanh
  G_prob = tanh(G_logp)
	#G_prob = reshape(G_prob,(size(G_prob,2),28,28))
	return G_prob/size(Z,2)
end


function samplenoise(bs,sizeZ,atype;winit=0.1)
  res = randn(sizeZ,bs)*winit
  return convert(atype,res)
end


function minibatch(x, y, batchsize; atype=Array{Float32}, xrows=784, yrows=10, xscale=255/2)
    xbatch(a)=convert(atype, reshape(a./xscale-1, xrows, div(length(a),xrows)))
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



function initalize_weights_D(x,D,atype)
  #= dims
  d1: (100,784)
  d2: (100,1)
  d3: (1,100)
  d4: (1,1)
  =#
  println("Initializing Discriminator weights.\n")
  Dnet = Any[]
  tmp=x
  for (i,j) in enumerate([D..., 1])
      push!(Dnet,xavier(j,tmp))
      push!(Dnet,zeros(j,1))
      tmp = j
  end
  #convert to Knet array or Array{Float32}
  return map(x->convert(atype, x), Dnet)
end


#will generate Gdim amount of images
function initalize_weights_G(x,z,G,atype)
  println("Initializing Generator weights.\n")
  #= dims
  g1: (100,784)
  g2: (100,100)
  g3: (1,100)
  g4: (1,1)
  =#

  Gnet = Any[]
  tmp=z
  for (i,j) in enumerate([G..., x])
      push!(Gnet,xavier(j,tmp))
      push!(Gnet,zeros(j,1))
      tmp = j
  end
  return map(x->convert(atype, x), Gnet)
end


  function loaddata()
      info("Loading MNIST...")
      gzload("train-images-idx3-ubyte.gz")[17:end],
      gzload("t10k-images-idx3-ubyte.gz")[17:end],
      gzload("train-labels-idx1-ubyte.gz")[9:end],
      gzload("t10k-labels-idx1-ubyte.gz")[9:end]
  end

  function gzload(file; path=Knet.dir("data",file), url="http://yann.lecun.com/exdb/mnist/$file")
      isfile(path) || download(url, path)
      f = gzopen(path)
      a = read(f)
      close(f)
      return(a)
  end


main(ARGS)
