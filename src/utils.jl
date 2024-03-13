export stop_gradient

stop_gradient(x) = ChainRulesCore.ignore_derivatives(x)
function stop_gradient(n::ForwardDiff.Dual{T}) where {T}
    ForwardDiff.Dual{T}(ForwardDiff.value(n), ForwardDiff.partials(1))
end
stop_gradient(n::Array{<:ForwardDiff.Dual}) = stop_gradient.(n)
