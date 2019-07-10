export rfactor!,
       calldeps,
       opcouple,
       init_IC,
       process

"""
    calldeps(N::Int, skipfirst::Bool=true)

If `skipfirst` == true, return the call dependency data structure
for the coupled integration of `N+1` dynamical systems, where the
first is given by a nonlinear equation and the remaining `N` solve
`N` linearised equations in tangent space. When `skipfirst` == false
the dependency data structure is simpler as assumes a set of `N`
independent linearised system. This last case can be used with
time integration using an `AbstractStorage`.
"""
function calldeps(N::Int, skipfirst::Bool=true)
    if skipfirst
        return CallDependency((1, ), [(1, i) for i = 2:N+1]...)
    else
        return CallDependency([(i, ) for i = 1:N]...)
    end
end

"""
    init_IC(N::Int, x::X) where {X<:AbstractVector}

Construct an array `xs` of length `N`, with elements of type `X`
similar to `x` and spaning the first `N-1` canonical directions
of the vector space where `x` belongs to.

Example:
    julia> init_IC(3, zeros(3))
    [1.0, 0.0, 0.0]
    [0.0, 1.0, 0.0]
    [0.0, 0.0, 1.0]

This function can be use to initialise the integration of the
coupled nonlinear/linearised equations.
"""
function init_IC(N::Int, x::X) where {X<:AbstractVector}
    xs = X[]
    for i = 1:N
        xi = similar(x); xi .= 0; xi[i] = 1
        xi ./= norm(xi)
        push!(xs, xi)
    end
    return xs
end

"""
    opcouple(N::Int, gN, gL)

Create a `Coupled` object packing the operator `gN` together
with `N` identical copies of the operator `gL`.
"""
opcouple(N::Int, gN, gL) = couple(gN, ntuple(i->deepcopy(gL), N)...)

"""
    opcouple(N::Int, gL)

Create a `Coupled` object packing `N` identical copies
of the operator `gL`.
"""
opcouple(N::Int, gL) = opcouple(N-1, gL, gL)

"""
    rfactor!(xs::Coupled{M}, skipfirst::Bool) where {M}

Two-pass Gram-Schmidt orthogonalisation of tangent vectors
stored in a `Coupled` object. If `skipfirst` is true, then
the first element of `xs` is the nonlinear state, and the
entries from `2:M` are tangent vectors. This function
modifies the tangent vectors in place and returns the upper
triangular factor `R`, from the `QR` factorisation of the
matrix `A` formed with the input tangent vectors as columns.
The size of the output is `M x M` is `skipfirst` is false
and `M-1 x M-1` is it is true.
"""
function rfactor!(xs::Coupled{M}, skipfirst::Bool) where {M}
    N, Δ = skipfirst == true ? (M-1, 1) : (M, 0)
    R = zeros(N, N)
    for i = 1:N
        for _ in (1, 2)
            for j in 1:i-1
                r = dot(xs[i+Δ], xs[j+Δ])
                xs[i+Δ] .-= r.*xs[j+Δ]
                R[j, i] += r
            end
        end
        R[i, i] = norm(xs[i+Δ])
        xs[i+Δ] ./= R[i, i]
    end
    return R
end

"""
    process(R::AbstractVector{<:NTuple{M, S}}, dt::Real) where {M, S}

Process data organised as an array of `M`-tuples in a `M`-tuple of arrays.
"""
function process(R::AbstractVector{<:NTuple{M, S}},
                ts::AbstractVector{<:Real}) where {M, S}
    _R = reinterpret(S, R)
    dts = vcat(1.0, diff(ts))
    ntuple(M) do i
        log.(_R[i:M:end])./dts
    end
end
