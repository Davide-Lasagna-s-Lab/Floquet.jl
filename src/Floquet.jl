"""
    Floquet

Floquet exponents, Floquet vectors and their adjoints of periodic and relative
periodic orbits, with a matrix-free periodic eigendecomposition along an orbit
stored with Flows.jl. The package exports one function, [`floquet`](@ref).
"""
module Floquet

import LinearAlgebra: dot,
                      norm,
                      mul!,
                      eigvals,
                      svd,
                      diag,
                      I,
                      UpperTriangular

using Flows
using Printf

export floquet

include("utils.jl")
include("eigendecomposition.jl")

end
