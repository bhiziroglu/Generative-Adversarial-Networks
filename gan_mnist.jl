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
          ("--epochs"; arg_type=Int; default=100)
          ("--batchsize"; arg_type=Int; default=100)
          ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
          ("--lr"; arg_type=Float32; default=Float32(0.3))
          ("--gencnt"; arg_type=Int; default=100; help="Number of images that generator function creates.")
          #try to keep gencnt with the same size as Z.
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
      Dnetopt = map(x->Adam(), Dnet)
      Gnetopt = map(x->Adam(), Gnet)

    @time for epoch=1:o[:epochs]
      shuffle!(trn)
      lossD = 0
      lossG = 0
      for i = 1:length(trn)
        x = trn[i][1]
        #train D
      ohG = onehotG(x,atype);
      ohD = onehotD(x,sizeZ,atype);
      for i=1:1
          lossD += D_loss(Dnet,sizeZ,x,ohD,o,Gnet,atype)
          Dnet = trainD(Dnet,Gnet,x,sizeZ,ohD,Dnetopt,o,atype)
        end

        #train G
        for i=1:1
          lossG += G_loss(Gnet,Dnet,sizeZ,ohG,atype,o)
          Gnet = trainG(Gnet,Dnet,sizeZ,ohG,Gnetopt,o,atype)
        end
      end
      @printf("epoch: %d loss[D]: %g loss[G]: %g\n",epoch,lossD/length(trn),lossG/length(trn))
      print_output(epoch,Gnet,sizeZ,atype,o)
    end

    end

  #one-hot vector for generative model
  function onehotG(x,atype)
    onehot = zeros(2,size(x,2));
    for i=1:size(x,2)
      onehot[1,i]=1
    end

    return convert(atype,onehot)
  end

  #one-hot
  function onehotD(x,sizeZ,atype)
    onehot_Real = zeros(2,size(x,2));
    onehot_Fake = zeros(2,sizeZ);
    for i=1:sizeZ
      onehot_Fake[2,i] = 1
    end

    for i=1:size(x,2)
      onehot_Real[1,i]=1
    end

    onehot = hcat(onehot_Real,onehot_Fake)
    return convert(atype,onehot)
  end


  function trainG(Gnet,Dnet,sizeZ,onehot,Gnetopt,o,atype)
      g = G_lossgradient(Gnet,Dnet,sizeZ,onehot,atype,o)
      for i=1:length(Gnet)
        update!(Gnet[i],g[i],Gnetopt[i])
      end
      return Gnet
  end


  function trainD(Dnet,Gnet,x,sizeZ,onehot,Dnetopt,o,atype)
      g = D_lossgradient(Dnet,sizeZ,x,onehot,o,Gnet,atype)
      for i=1:length(Dnet)
        update!(Dnet[i],g[i],Dnetopt[i])
      end
      return Dnet
  end

    #TODO - figure out how to print (without choosing randomly)
  function print_output(epoch,Gnet,sizeZ,atype,o)
      gg = generator(Gnet,sizeZ,o[:batchsize],atype,o[:gencnt])
      gg = (gg+1)/2
      gg = min(1,max(0,gg))
      gg = convert(Array{Float64},gg)
      gg = gg[:,1:1]
      gg = reshape(gg,(28,28))
      save(@sprintf("output%d.png",epoch),gg)
  end

    #discriminator loss
    function D_loss(Dnet,sizeZ,x,onehot,o,Gnet,atype)
    	G_fake = generator(Gnet,sizeZ,o[:batchsize],atype,o[:gencnt])
      G_real = x
      G = hcat(G_real,G_fake)
      D_fake = discriminator(Dnet,G,atype)
      D_logit = logp(D_fake)
      return -sum(D_logit.*onehot)/size(G,2)
    end

    D_lossgradient = grad(D_loss)

    #generator loss
    function G_loss(Gnet,Dnet,sizeZ,onehot,atype,o)
    	G_sample = generator(Gnet,sizeZ,o[:batchsize],atype,o[:gencnt])
    	D_fake = discriminator(Dnet,G_sample,atype) #fake prob
      G_logit = logp(D_fake)
      return -sum(G_logit.*onehot)/size(G_sample,2)
    end

    G_lossgradient = grad(G_loss)

    #returns a probability which tells whether the input image
    #is from the real dataset or a generated one
    function discriminator(Dnet,x,atype)
      D_h1 = tanh(Dnet[1] * x .+ Dnet[2] );
      D_h1 = dropout(D_h1,0.5,atype);
    	D_logit = Dnet[3] * D_h1 .+ Dnet[4];
    	D_prob = (D_logit)
      return D_prob
    end


    #it takes N-dimensional vector where N is an arbitrary number
    #N is the dimension of the Z vector
    #return a 784 dimensional MNIST image
    function generator(Gnet,sizeZ,bs,atype,gencnt)
      Z = samplenoise(gencnt,sizeZ,atype)
      G_h1 = tanh(Gnet[1] * Z .+ Gnet[2]);
      G_h1 = dropout(G_h1,0.5,atype);
    	G_logp = Gnet[3] * G_h1 .+ Gnet[4];
      G_prob = tanh(G_logp)
    	return G_prob
    end


    function dropout(h1,prob,atype)
        return h1 .* convert(atype,((randn(size(h1)).>prob) / (1-prob)))
    end

    function samplenoise(bs,sizeZ,atype;winit=1)
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
      println("Initializing Discriminator weights.")
      Dnet = Any[]
      for (i,j) in enumerate([D..., 2])
          push!(Dnet,xavier(j,x))
          push!(Dnet,zeros(j,1))
          x = j
      end
      #convert to Knet array or Array{Float32}
      return map(x->convert(atype, x), Dnet)
    end


    #will generate Gdim amount of images
    function initalize_weights_G(x,z,G,atype)
      println("Initializing Generator weights.")
      Gnet = Any[]
      for (i,j) in enumerate([G..., x])
          push!(Gnet,xavier(j,z))
          push!(Gnet,zeros(j,1))
          z = j
      end
      #return convert(atype,Gnet)
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
