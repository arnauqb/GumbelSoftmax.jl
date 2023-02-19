export sample_gumbel_softmax

function sample_gumbel(probs::CuArray, size; epsilon=1e-10)
    ret = CUDA.zeros(size...)
    rand!(ret)
    ret = -log.(-log.(ret .+ epsilon) .+ epsilon)
    return ret
end

function sample_gumbel(probs, size; epsilon=1e-10)
    ret = rand(size...)
    ret = -log.(-log.(ret .+ epsilon) .+ epsilon)
    return ret
end

function sample_gumbel_softmax(probs, tau; hard=true)
    logits = log.(probs .+ 1e-16)
    y = logits + sample_gumbel(probs, size(logits))
    y_soft = softmax(y / tau, dims=2)
    if hard
        y_hard = (y_soft .== maximum(y_soft, dims=2))
        ret = y_hard - Zygote.dropgrad(y_soft) + y_soft
    else
        ret = y_soft
    end
    return ret
end
