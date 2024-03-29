using Pkg
Pkg.activate(".")
using GumbelSoftmax, Flux, MLDatasets, Statistics, Zygote, ProgressMeter, PyPlot

device = gpu

function get_mnist()
    xtrain, ytrain = MNIST(:train)[:]
    xtest, ytest = MNIST(:test)[:]
    return xtrain, ytrain, xtest, ytest
end
xtrain, _, xtest, _ = get_mnist()
input_dim = size(xtrain, 1) * size(xtrain, 2)
xtrain_flat = reshape(xtrain, input_dim, size(xtrain, 3)) |> device

input_dim = 28^2
latent_dim = 30
categorical_dim = 10
encoder = Chain(
    Dense(28^2, 512, relu),
    Dense(512, 256, relu),
    Dense(256, latent_dim * categorical_dim, relu),
) |> device
decoder = Chain(
    Dense(latent_dim * categorical_dim, 256, relu),
    Dense(256, 512, relu),
    Dense(512, input_dim, sigmoid),
) |> device

function run_model(encoder, decoder, x, latent_dim, categorical_dim)
    q = encoder(x)
    #q = permutedims(q1, (2, 1))
    #q = reshape(q, :, latent_dim, categorical_dim)
    q_y = reshape(q, latent_dim, categorical_dim, :)
    z = sample_gumbel_softmax(logits=q_y, tau=0.5)
    z = reshape(z, latent_dim * categorical_dim, :)
    #z = permutedims(z, (2, 3, 1))
    #z = reshape(z, latent_dim * categorical_dim, :)
    return decoder(z), reshape(softmax(q_y, dims=2), latent_dim * categorical_dim, :)
end
z_decoded, z_soft = run_model(encoder, decoder, xtrain_flat[:, 1:2], latent_dim, categorical_dim);

##
function compute_loss(x, x_reconstructed, latent_z)
    bce = Flux.Losses.binarycrossentropy(x_reconstructed, x, agg=sum) ./ size(x, 2)
    log_ratio = log.(latent_z .* categorical_dim .+ 1e-10) 
    kld = mean(sum(latent_z .* log_ratio, dims=1))
    return bce + kld
end
compute_loss(xtrain_flat[:, 1:2], z_decoded, z_soft)

function train(encoder, decoder, xtrain, nepochs)
    loader = Flux.DataLoader((xtrain), batchsize=64, shuffle=true)
    optim = Flux.Optimise.Adam(1e-3)
    losses = []
    trainable_params = Flux.params(encoder, decoder)
    @showprogress for epoch in 1:nepochs
        for x in loader
            loss, back = Flux.pullback(trainable_params) do
                z_decoded, z_soft = run_model(encoder, decoder, x, latent_dim, categorical_dim)
                compute_loss(x, z_decoded, z_soft)
            end
            gradients = back(1f0)
            Flux.Optimise.update!(optim, trainable_params, gradients)
            push!(losses, loss)
        end
    end
    return losses
end

losses = train(encoder, decoder, xtrain_flat, 1);

##
fig, ax = plt.subplots()
ax.plot(losses)
fig
##

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
xt = xtest[:, :, 6]
ax[1].imshow(transpose(xt))
xt_flat = reshape(xt, input_dim, 1) |> device
x_reconsructed = run_model(encoder, decoder, xt_flat, latent_dim, categorical_dim)[1]
x_reconsructed = reshape(x_reconsructed |> cpu, 28, 28)
ax[2].imshow(transpose(x_reconsructed |> cpu))
fig
##
