for p in ("Knet","ArgParse","Compat","GZip","Images","JLD")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet
using ArgParse
using Compat, GZip
using Images
using JLD


function main(args)
    s = ArgParseSettings()
    s.description = "Implementation of the paper Generative Adversarial Networks [https://arxiv.org/abs/1406.2661]";
    s.exc_handler=ArgParse.debug_handler

    @add_arg_table s begin
      ("--epochs"; arg_type=Int; default=100)
      ("--batchsize"; arg_type=Int; default=100)
      ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
      ("--lr"; arg_type=Float32; default=Float32(0.1))
      ("--z"; arg_type=Int; default=784)
      ("--gencnt"; arg_type=Int; default=100; help="Number of images that generator function creates.")
      #try to keep gencnt with the same size as Z.
      ("--print"; default=true ; help="Set false to turn off creating output images")
    end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  global o = parse_args(args, s; as_symbols=true)
  global atype = eval(parse(o[:atype]))
  info("GAN Started...")
  println(o)


  #Download MNIST dataset
  (xtrn,xtst,ytrn,ytst)=loaddata()

  trn = minibatch(xtrn, ytrn, o[:batchsize];atype=atype)
  #tst = minibatch(xtst, ytst, o[:batchsize];atype=atype)


  if(ispath("Dnet.jld") && ispath("Gnet.jld"))
    info("Weights found")
    info("Using pre-trained weights")
    Dnet = load("Dnet.jld","Dnet")
    Gnet = load("Gnet.jld","Gnet")
  else
    Dnet = initalize_weights_D();
    Gnet = initalize_weights_G();
  end

  #ADAM
  Dnetopt = map(x->Adam(;lr=0.0002,beta1=0.5), Dnet)
  Gnetopt = map(x->Adam(;lr=0.0002,beta1=0.5), Gnet)


  #output to log file
  fh = open("log.txt","w")



@time for epoch=1:o[:epochs]
  shuffle!(trn)
  lossD = 0
  lossG = 0
  total = length(trn)
  for i = 1:total
    x = trn[i][1]

    #train D
    for i=1:1
      lossD += D_loss(Dnet,x,Gnet)
      trainD!(Dnet,Gnet,x,Dnetopt)
    end

    #train G
    for i=1:1
      lossG += G_loss(Gnet,Dnet)
      trainG!(Gnet,Dnet,Gnetopt)
    end

  end
  @printf("epoch: %d loss[D]: %g loss[G]: %g\n",epoch,lossD/total,lossG/total)
  write(fh,"epoch: $epoch loss[D]: $lossD loss[G]: $lossG\n")
  print_output(epoch,Gnet)
  save_weights(Dnet,Gnet)
end

  #close file IO
  close(fh)
end


function save_weights(Dnet,Gnet)
  save("Dnet.jld","Dnet",Dnet)
  save("Gnet.jld","Gnet",Gnet)
end

function print_output(epoch,Gnet)
  Z = samplenoise()
  G_h1 = tanh(Gnet[1] * Z .+ Gnet[2])
  G_logp = (Gnet[3] * G_h1 .+ Gnet[4])
  G_sample = sigm(G_logp)
  G_sample = convert(Array{Float64},G_sample)
  out = makegrid(G_sample)
  save(@sprintf("output%d.png",epoch),out)
end


function trainG!(Gnet,Dnet,Gnetopt)
  g = G_lossgradient(Gnet,Dnet)
  for i=1:length(Gnet)
    update!(Gnet[i],g[i],Gnetopt[i])
  end
  return Gnet
end


function trainD!(Dnet,Gnet,x,Dnetopt)
  g = D_lossgradient(Dnet,x,Gnet)
  for i=1:length(Dnet)
    update!(Dnet[i],g[i],Dnetopt[i])
  end
  return Dnet
end



#discriminator loss
function D_loss(Dnet,x,Gnet)
  Z = samplenoise()
	G_fake = generator(Gnet,Z)
  ep = 1e-10
  G_real = x
  D_real = log(discriminator(Dnet,G_real)+ep)/2
  D_fake = log(1-discriminator(Dnet,G_fake)+ep)/2
  -sum(D_real + D_fake)/(size(D_real,2)+size(D_fake,2))
end

D_lossgradient = grad(D_loss)

#generator loss
function G_loss(Gnet,Dnet)
  Z = samplenoise()
	G_sample = generator(Gnet,Z)
  ep=1e-10
	D_fake = discriminator(Dnet,G_sample) #fake prob
  G_logit = log(D_fake)/2
  -sum(G_logit)/size(G_logit,2)
end

G_lossgradient = grad(G_loss)

#returns a probability which tells whether the input image
#is from the real dataset or a generated one
function discriminator(Dnet,x)
  art_noise =  convert(KnetArray{Float32},randn(size(x))*0.001)
  x = tanh(x+art_noise) #adding some artificial noise
  D_h1 = tanh(Dnet[1] * x .+ Dnet[2])
	D_logit = (Dnet[3] * D_h1 .+ Dnet[4])
	sigm(D_logit)
end


function generator(Gnet,sampnoise)
  Z = sampnoise
  G_h1 = tanh(Gnet[1] * Z .+ Gnet[2])
  G_h1 = dropout(G_h1,0.5,atype);
	G_logp = (Gnet[3] * G_h1 .+ Gnet[4])
  tanh(G_logp)
end


function dropout(h1,prob,atype)
    return h1 .* convert(atype,((randn(size(h1)).>prob) / (1-prob)))
end

function samplenoise()
  res = randn(o[:z],o[:batchsize])
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



function initalize_weights_D()
  println("Initializing Discriminator weights.")
  winit=0.002
  Dnet = Array(Any,4)
  Dnet[1] = randn(128,784)*winit
  Dnet[2] = zeros(128,1)
  Dnet[3] = randn(2,128)*winit
  Dnet[4] = zeros(2,1)
  return map(x->convert(atype, x), Dnet)
end


function initalize_weights_G()
  println("Initializing Generator weights.")
  winit=0.002
  Gnet = Array(Any,4)
  Gnet[1] = randn(128,784)*winit
  Gnet[2] = zeros(128,1)
  Gnet[3] = randn(784,128)*winit
  Gnet[4] = zeros(784,1)
  return map(x->convert(atype, x), Gnet)
end

function makegrid(y; gridsize=[10,10], scale=2.0, shape=(28,28))
    y = reshape(y, shape..., size(y,2))
    y = map(x->y[:,:,x]', [1:size(y,3)...])
    shp = map(x->Int(round(x*scale)), shape)
    y = map(x->Images.imresize(x,shp), y)
    gridx, gridy = gridsize
    outdims = (gridx*shp[1]+gridx+1,gridy*shp[2]+gridy+1)
    out = zeros(outdims...)
    for k = 1:gridx+1; out[(k-1)*(shp[1]+1)+1,:] = 1.0; end
    for k = 1:gridy+1; out[:,(k-1)*(shp[2]+1)+1] = 1.0; end

    x0 = y0 = 2
    for k = 1:length(y)
        x1 = x0+shp[1]-1
        y1 = y0+shp[2]-1
        out[x0:x1,y0:y1] = y[k]

        y0 = y1+2
        if k % gridy == 0
            x0 = x1+2
            y0 = 2
        else
            y0 = y1+2
        end
    end

    return convert(Array{Float64,2}, map(x->isnan(x)?0:x, out))
end

function loaddata()
    info("Loading MNIST...")
    gzload("train-images-idx3-ubyte.gz")[17:end],
    gzload("t10k-images-idx3-ubyte.gz")[17:end],
    gzload("train-labels-idx1-ubyte.gz")[9:end],
    gzload("t10k-labels-idx1-ubyte.gz")[9:end]
end

function gzload(file; path="$file", url="http://yann.lecun.com/exdb/mnist/$file")
  isfile(path) || download(url, path)
  f = gzopen(path)
  a = @compat read(f)
  close(f)
  return(a)
end


main(ARGS)
