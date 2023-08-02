# GumbelSoftmax

[![Build Status](https://github.com/arnauqb/GumbelSoftmax.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/arnauqb/GumbelSoftmax.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/arnauqb/GumbelSoftmax.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/arnauqb/GumbelSoftmax.jl)

This package implements the [Gumbel-Softmax reparametrization trick](https://arxiv.org/abs/1611.01144) in Julia using [Zygote](https://github.com/FluxML/Zygote.jl).

# Usage

Let's suppose we want to sample from 4 Categorical variables each with 3 possible outcomes. We can input the probabilities like this:

```julia
julia> p = [[0.1, 0.2, 0.3, 0.4] [0.5, 0.6, 0.1, 0.3] [0.4, 0.2, 0.6, 0.3]]
4×3 Matrix{Float64}:
 0.1  0.5  0.4
 0.2  0.6  0.2
 0.3  0.1  0.6
 0.4  0.3  0.3
```
In this case the note that the rows add to 1, but it is not necessary to have the matrix normalized. We can then sample in a differentiable way like this:

```julia
julia> using GumbelSoftmax
julia> result = sample_gumbel_softmax(p, 0.1; hard=true)
4×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  0.0  1.0
 1.0  0.0  0.0
 0.0  1.0  0.0
 ```
 where 0.1 corresponds to the temperature parameter, and `hard=true` specified whether we want hard samples (ie 0 or 1). We can then calculate the gradient using the standard Zygote interface. Note that we can only calculate gradients, so we will suppose we are interested in the sum of results in the second column:

```julia
using Zygote
function to_derive(p)
    result = sample_gumbel_softmax(p, 0.1; hard=true)
    return sum(result[:,2])
end

grad_value = gradient(to_derive, p)
([-1.8664458645494402e-16 1.6225865540064885e-7 -2.0282331979907546e-7; -2.152607762114983e-16 8.83806125520096e-11 -2.651416223952525e-10; -9.333173809814049e-18 5.633448015616505e-9 -9.38907997936165e-10; -1.1976417933816025e-8 0.009956664887966756 -0.009956648919416302],)
```


