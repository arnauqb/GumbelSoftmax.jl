export sample_gumbel_softmax

Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)

function sample_gumbel(probs::CuArray, size; epsilon=1e-10)
    ret = CUDA.zeros(size...)
    rand!(ret)
    ret = -log.(-log.(ret .+ epsilon) .+ epsilon)
    return ret
end

function sample_gumbel(probs, size; epsilon=1e-10)
    ret = rand(Float32, size...)
    ret = -log.(-log.(ret .+ epsilon) .+ epsilon)
    return ret
end

function sample_gumbel_softmax(; probs = nothing, logits = nothing, tau = 0.1, hard = true, epsilon = 1e-10)
    epsilon = Float32(epsilon)
    if logits === nothing
        logits = log.(probs .+ epsilon)
    end
    y = logits + sample_gumbel(logits, size(logits), epsilon = epsilon)
    y_soft = softmax(y / tau, dims = 2)
    if hard
        y_hard = (y_soft .== maximum(y_soft, dims = 2))
        ret = y_hard - stop_gradient(y_soft) + y_soft
    else
        ret = y_soft
    end
    return ret
end