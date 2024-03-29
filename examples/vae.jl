##
using Distributions, GumbelSoftmax, Flux, MLDatasets, Statistics, ProgressMeter, PyPlot

##
# parameters

device = gpu
latent_dim = 30
categorical_dim = 10

function get_mnist()
    xtrain, ytrain = MNIST(:train)[:]
    xtest, ytest = MNIST(:test)[:]
    return xtrain, ytrain, xtest, ytest
end
xtrain, _, xtest, _ = get_mnist()
input_dim = size(xtrain, 1) * size(xtrain, 2)
xtrain_flat = reshape(xtrain, input_dim, size(xtrain, 3)) |> device

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

##
function run_model(encoder, decoder, x, latent_dim, categorical_dim)
    q = encoder(x)
    q_unflatten = reshape(q, latent_dim, categorical_dim, :)
    z = sample_gumbel_softmax(logits=q_unflatten, tau=0.5)
    #z = sample_rao_gumbel_softmax(logits=q_unflatten |> cpu, tau=0.5, k=1) |> device
    z = reshape(z, latent_dim * categorical_dim, :)
    return decoder(z), reshape(softmax(q_unflatten, dims=2), latent_dim * categorical_dim, :)
end

function compute_loss(x, x_reconstructed, latent_z)
    bce = Flux.Losses.binarycrossentropy(x_reconstructed, x, agg=sum) ./ size(x, 2)
    log_ratio = log.(latent_z .* categorical_dim .+ 1e-10) 
    kld = mean(sum(latent_z .* log_ratio, dims=1))
    return bce + kld
end

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
##

losses = train(encoder, decoder, xtrain_flat, 10);

##
# plot loss
fig, ax = plt.subplots()
ax.plot(losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
fig.savefig("examples/img/losses.png", bbox_inches="tight")
##

# plot reconstruction examples
n_examples = 10
fig, ax = plt.subplots(n_examples, 2, figsize=(2, 5))
# plot original images on the left and reconstructed images on the right
for i in 1:n_examples
    xt = xtest[:, :, i]
    ax[i, 1].imshow(transpose(xt), cmap="gray")
    xt_flat = reshape(xt, input_dim, 1) |> device
    x_reconsructed = run_model(encoder, decoder, xt_flat, latent_dim, categorical_dim)[1]
    x_reconsructed = reshape(x_reconsructed |> cpu, 28, 28)
    ax[i, 2].imshow(transpose(x_reconsructed), cmap="gray")
end
# set titles
ax[1, 1].set_title("Original", fontsize=8)
ax[1, 2].set_title("Reconstructed", fontsize=8)
# remove axis
for a in ax
    a.axis("off")
end
plt.subplots_adjust(wspace=0.1, hspace=0.01)
fig.savefig("examples/img/reconstructed.png", bbox_inches="tight")
fig
##

##
# plot sampled examples
# sample from categorical distribution and decode
n_samples = 64
M = n_samples * latent_dim
samples = rand(Categorical(0.1 ./ ones(categorical_dim)), M)
samples_oh = Float32.(reduce(hcat, Flux.onehot.(samples, Ref(1:categorical_dim))))
samples_oh = reshape(samples_oh, latent_dim * categorical_dim, n_samples) |> device
samples_decoded = decoder(samples_oh)
samples_decoded = reshape(samples_decoded |> cpu, 28, 28, n_samples)


fig, ax = plt.subplots(8, 8, figsize=(8, 8))
for index in 1:n_samples
    i = Int(ceil(index / 8))
    j = index % 8
    if j == 0
        j = 8
    end
    ax[i, j].imshow(transpose(samples_decoded[:,:,index]), cmap="gray")
    ax[i, j].axis("off")
end
fig.savefig("examples/img/generated.png", bbox_inches="tight")
fig
#
