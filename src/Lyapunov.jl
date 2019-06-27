module Lyapunov

import LinearAlgebra: dot, norm
import Flows

export gsortho!,
       calldeps,
       opcouple,
       init_IC,
       process

"""
    calldeps(N::Int)

Define the call dependency data structure for the coupled integration of 
`N` dynamical systems, where the first is given by a nonlinear equation and 
the remaining `N-1` solve `N-1` linearised equations in tangent space.
"""
calldeps(N::Int) = Flows.CallDependency((1, ), [(1, i) for i = 2:N]...)

"""
    init_IC(N::Int, x::X) where {X<:AbstractVector}

Construct an array `xs` of length `N`, with elements of type `X`.
The first element is `x`, while the remaining elements span the first 
`N-1` canonical directions of the vector space where `x` belongs to. 

Example:
    julia> init_IC(3, zeros(3))
    [0.0, 0.0, 0.0]
    [1.0, 0.0, 0.0]
    [0.0, 1.0, 0.0]

This function can be use to initialise the integration of the 
coupled nonlinear/linearised equations.
"""
function init_IC(N::Int, x::X) where {X<:AbstractVector}
    xs = X[x]
    for i = 1:N-1
        xi = similar(x); xi .= 0; xi[i] = 1
        xi ./= norm(xi)
        push!(xs, xi)
    end
    return xs
end

"""
    opcouple(N::Int, gN, gL)

Create a `Flows.Coupled` object packing the operator `gN` together 
with `N-1` identical copies of the operator `gL`.
"""
opcouple(N::Int, gN, gL) = Flows.couple(gN, ntuple(i->deepcopy(gL), N-1)...)

"""
    gsortho!(xs::Flows.Coupled{N}) where {N}

Two-pass Gram-Schmidt orthogonalisation of `N-1` tangent vectors
stored in a `Flows.Coupled` object, where `xs[1]` is the nonlinear
state, and the entries `2:N` are the tangent vectors. This function 
modifies the tangent vectors in place and returns the diagonal of 
the matrix `R`, from the thin `QR` factorisation of the matrix `A`
formed with the tangent vectors as columns.
"""
@generated function gsortho!(xs::Flows.Coupled{N}) where {N}
    quote
        # return diagonal of the `R` factor as an `N-1` tuple
        # this makes the code a bit faster, since a vector of 
        # tuples can be stored more efficiently, with less 
        # allocations of heap memory going on.
        return Base.Cartesian.@ntuple $(N-1) i -> begin
            for _ in (1, 2)
                for j in 1:i-1
                    r = dot(xs[i+1], xs[j+1])
                    xs[i+1] .-= r.*xs[j+1]
                end
            end
            rii = norm(xs[i+1])
            xs[i+1] ./= rii
            rii
        end
    end
end

"""
    process(R::AbstractVector{<:NTuple{M, T}}, dt::Real) where {M, T}

Process data organised as an array of `M`-tuples in a `M`-tuple of arrays.
"""
function process(R::AbstractVector{<:NTuple{M, T}}, dt::Real) where {M, T}
    _R = reinterpret(T, R)
    ntuple(M) do i
        log.(_R[i:M:end])./dt
    end
end

end