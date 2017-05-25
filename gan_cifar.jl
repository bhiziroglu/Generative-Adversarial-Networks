    for p in ("Knet","ArgParse","Images","BinDeps","JLD")
        Pkg.installed(p) == nothing && Pkg.add(p)
    end

    using Knet
    using ArgParse
    using AutoGrad
    using GZip
    using Compat
    using Images
    using ImageMagick
    using BinDeps
    using MAT
    using Base
    using JLD


function main(args)
        s = ArgParseSettings()
        s.description = "Implementation of the paper Generative Adversarial Networks [https://arxiv.org/abs/1406.2661] \nUsing Knet Library in Julia";
        s.exc_handler=ArgParse.debug_handler

        @add_arg_table s begin
          ("--epochs"; arg_type=Int; default=20)
          ("--batchsize"; arg_type=Int; default=128)
          ("--noisedim"; arg_type=Int; default=100)
          ("--log"; default=true; help="Save the losses to log file")
          ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
          ("--lr"; arg_type=Float32; default=Float32(0.3))
          ("--gencnt"; arg_type=Int; default=100; help="Number of images that generator function creates.")
          #try to keep gencnt with the same size as Z.
          ("--print"; default=true ; help="Set false to turn off creating output images")
        end

      isa(args, AbstractString) && (args=split(args))
      global o = parse_args(args, s; as_symbols=true)
      global atype = eval(parse(o[:atype]))
      info("GAN Started...")
      info(now())
      println(o)
      info("Loading CIFAR-10...")
      (xtrn,xtst)=traindata()
      info("CIFAR-10 Loaded")

      trn = 2*(minibatch(xtrn,o[:batchsize];atype=atype).-0.5) #normalization (-1,1)
      #32 * 32 * 3 * bs
      #trn = 2*(trn.-0.5)
      if(ispath("Dnet.jld") && ispath("Gnet.jld"))
        info("Weights found")
        info("Using pre-trained weights")
        Dnet = load("Dnet.jld","Dnet")
        Gnet = load("Gnet.jld","Gnet")
      else
        Dnet = initalize_weights_D(atype);
        Gnet = initalize_weights_G(atype);
      end

      #ADAM
      Dnetopt = map(x->Adam(lr=0.0002, beta1=0.5), Dnet)
      Gnetopt = map(x->Adam(lr=0.0002, beta1=0.5), Gnet)

    fh = open("log.txt","w")
    total = Int(floor(length(trn)))
    @time for epoch=1:o[:epochs]
      shuffle!(trn)
      lossG = 0
      lossD = 0
      @time for i = 1:total
        x = trn[i]

        for k = 1:1 #k value used 1. least cost
          trainD!(Dnet,Gnet,Dnetopt,x)
        end

        for k = 1:1 #k value used 1. least cost
          trainG!(Gnet,Dnet,Gnetopt)
        end

        lossD += D_loss(Dnet,Gnet,x)
        lossG += G_loss(Gnet,Dnet)
      end

    #if(o[:log]==true)
    write(fh,"epoch: $epoch loss[D]: $lossD loss[G]: $lossG\n")
    #end
    save_weights(Dnet,Gnet)
    print_output(epoch,Gnet)
    @printf("epoch: %d loss[D]: %g loss[G]: %g\n",epoch,lossD/total,lossG/total)
  end

close(fh)

end

function save_weights(Dnet,Gnet)
  save("Dnet.jld","Dnet",Dnet)
  save("Gnet.jld","Gnet",Gnet)
end


function trainG!(Gnet,Dnet,Gnetopt)
  g = G_lossgradient(Gnet,Dnet)
  for i in 1:length(Gnet)
    update!(Gnet[i],g[i],Gnetopt[i])
  end
end


function trainD!(Dnet,Gnet,Dnetopt,x)
  g = D_lossgradient(Dnet,Gnet,x)
  for i in 1:length(Dnet)
    update!(Dnet[i],g[i],Dnetopt[i])
  end
end


function print_sample(Gnet)
  noise = sampnoise()
  x = Gnet[1]*noise .+ Gnet[2];
  x = tanh(batchnorm(reshape(x, 2, 2, 1024,o[:batchsize])));
  x = tanh(batchnorm(deconv4(Gnet[3],x;padding=2, stride=2) .+ Gnet[4]));
  x = tanh(batchnorm(deconv4(Gnet[5],x;padding=2, stride=2) .+ Gnet[6]));
  x = tanh(batchnorm(deconv4(Gnet[7],x;padding=2, stride=2) .+ Gnet[8]));
  x = (deconv4(Gnet[9],x;padding=2, stride=2) .+ Gnet[10]);
  sample = sigm(x); #RGB pixels are in range (0,1) - tanh() causes problems
  return sample
end

#TODO - figure out how to print (without choosing randomly)
function print_output(epoch,Gnet)
  for i=1:10
    gg = print_sample(Gnet)
    out = makegrid(gg)
    save(@sprintf("output%d_%d.png",epoch,i),out)
  end
end


function sampnoise()
    return convert(atype,randn(o[:noisedim],o[:batchsize]))
end
    #discriminator loss
function D_loss(Dnet,Gnet,real)
  noise = sampnoise()
  G_sample = generator(Gnet,noise)
  D_real = discriminator(Dnet,real)
  ep = 1e-10
  D_logit = log(D_real + ep)/2
  real = -sum(D_logit)/size(D_logit,2)
  G_fake = discriminator(Dnet,G_sample)
  G_logit = log(1-G_fake+ep)/2
  fake = -sum(G_logit)/size(G_logit,2)
  real+fake
end

D_lossgradient = grad(D_loss)

#generator loss
function G_loss(Gnet,Dnet)
  noise = sampnoise()
  ep = 1e-10
  G_sample = generator(Gnet,noise)
  G_logit = discriminator(Dnet,G_sample)
  G_h = log(G_logit + ep)/2
  -sum(G_h)/size(G_h,2)
end

G_lossgradient = grad(G_loss)

function initalize_weights_G(atype)
  winit=0.02
  depths = [1024,512,256,128,3];
  println("Initializing Generator weights.")
  Gnet = Array(Any,10);
  Gnet[1] = xavier(2*2*depths[1],100)*winit;
  Gnet[2]= zeros(2*2*depths[1],1);
  Gnet[3] = randn(6,6,depths[2],depths[1])*winit;
  Gnet[4] = zeros(1,1,depths[2],1);
  Gnet[5] = randn(6,6,depths[3],depths[2])*winit;
  Gnet[6] = zeros(1,1,depths[3],1);
  Gnet[7] = randn(6,6,depths[4],depths[3])*winit;
  Gnet[8] = zeros(1,1,depths[4],1);
  Gnet[9] = randn(6,6,depths[5],depths[4])*winit;
  Gnet[10] = zeros(1,1,depths[5],1);
  return map(x->convert(atype, x), Gnet)
end

function generator(Gnet,sampnoise)
  sampnoise = convert(KnetArray{Float32},sampnoise)
  x = Gnet[1]*sampnoise .+ Gnet[2];
  x = reshape(x, 2, 2, 1024, o[:batchsize]);
  x = tanh(batchnorm(x));
  x = tanh(batchnorm(deconv4(Gnet[3],x;padding=2, stride=2) .+ Gnet[4]));
  x = tanh(batchnorm(deconv4(Gnet[5],x;padding=2, stride=2) .+ Gnet[6]));
  x = tanh(batchnorm(deconv4(Gnet[7],x;padding=2, stride=2) .+ Gnet[8]));
  x = deconv4(Gnet[9],x;padding=2, stride=2) .+ Gnet[10]
  tanh(x)
end

function discriminator(Dnet,x)
    art_noise = convert(KnetArray{Float32},randn(size(x))*0.1)
    x = x + art_noise #artificial noise
    x = tanh(x)
    x = LeakyReLU(conv4(Dnet[1],x;padding=2, stride=2) .+ Dnet[2])
    x = LeakyReLU(batchnorm(conv4(Dnet[3],x;padding=2, stride=2) .+ Dnet[4]))
    x = LeakyReLU(batchnorm(conv4(Dnet[5],x;padding=2, stride=2) .+ Dnet[6]))
    x = LeakyReLU(batchnorm(conv4(Dnet[7],x;padding=2, stride=2) .+ Dnet[8]))
    x = mat(x)
    sigm(Dnet[9]*x .+ Dnet[10])
end


function LeakyReLU(x;leak=0.2)
  return max(x*leak,x)
end


function initalize_weights_D(atype)
  winit=0.02
  depths = [128,256,512,1024]
  println("Initializing Discriminator weights.")
  Dnet = Array(Any,10)
  Dnet[1] = randn(6,6,3,depths[1])*winit
  Dnet[2] = zeros(1,1,depths[1],1)
  Dnet[3] = randn(6,6,depths[1],depths[2])*winit
  Dnet[4] = zeros(1,1,depths[2],1)
  Dnet[5] = randn(6,6,depths[2],depths[3])*winit
  Dnet[6] = zeros(1,1,depths[3],1)
  Dnet[7] = randn(6,6,depths[3],depths[4])*winit
  Dnet[8] = zeros(1,1,depths[4],1)
  Dnet[9] = randn(1,2*2*depths[4])*winit;
  Dnet[10] = zeros(1,1)
 return map(x->convert(atype, x), Dnet)
end

function dropout(h1,prob,atype)
    return h1 .* convert(atype,((randn(size(h1)).>prob) / (1-prob)))
end



function minibatch(x,batchsize; atype=Array{Float32})
  data = Any[]
  for i=1:batchsize:size(x,4)-batchsize+1
    j=i+batchsize-1
    push!(data,x[:,:,:,i:j])
  end
  return map(x->convert(atype, x),data)
end

function makegrid(y; gridsize=[10,10], scale=1.0, shape=(32,32))
    y = convert(Array{Float32},y)
    y = y[:,:,:,1:100]
    #y = reshape(y, 32,32,3,100)
    y = permutedims(y, [3,1,2,4])
    shp = map(x->Int(round(x*scale)), shape)
    t = Any[]
    for i in 1:100
      push!(t, y[:,:,:,i])
    end
    y = map(x->Images.colorview(RGB, x), t)
    gridx, gridy = gridsize


    out = zeros(3,331,331)
    out = Images.colorview(RGB, out)

    x0 = y0 = 1
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

    return out
end


const defdir = "datasets/cifar10"


function getdata(dir)
    mkpath(dir)
    info("Downloading CIFAR-10 dataset...")
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

#Batch normalization layer
function batchnorm(x;epsilon=1e-5)
  mu, sigma = nothing, nothing
  d = ndims(x) == 4 ? (1,2,4) : (2,)
  s = prod(size(x)[[d...]])
  mu = sum(x,d) / s
  sigma = sqrt(epsilon + (sum((x.-mu).*(x.-mu), d)) / s)
  (x.-mu) ./ sigma
end


# To be able to load/save KnetArrays:
if Pkg.installed("JLD") != nothing
    import JLD: writeas, readas
    type KnetJLD; a::Array; end
    writeas(c::KnetArray) = KnetJLD(Array(c))
    readas(d::KnetJLD) = KnetArray(d.a)
end

main(ARGS)
