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

stop_gradient(x) = ChainRulesCore.ignore_derivatives(x)
function stop_gradient(n::ForwardDiff.Dual{T}) where {T}
    ForwardDiff.Dual{T}(ForwardDiff.value(n), ForwardDiff.partials(1))
end
stop_gradient(n::Array{<:ForwardDiff.Dual}) = stop_gradient.(n)

function sample_gumbel_softmax(probs, tau; hard=true, epsilon=1e-10)
    logits = log.(probs .+ epsilon)
    y = logits + sample_gumbel(probs, size(logits), epsilon=epsilon)
    y_soft = softmax(y / tau, dims=2)
    if hard
        y_hard = (y_soft .== maximum(y_soft, dims=2))
        ret = y_hard - stop_gradient(y_soft) + y_soft
    else
        ret = y_soft
    end
    return ret
end
