"""
    gs!(xs::Coupled{M}, skipfirst::Bool=true) where {M}

Two-pass Gram-Schmidt orthogonalisation of tangent vectors
stored in a `Coupled` object. If `skipfirst` is true, then
the first element of `xs` is the nonlinear state, and the
entries from `2:M` are tangent vectors. This function
modifies the tangent vectors in place and returns the upper
triangular factor `R`, from the `QR` factorisation of the
matrix `A` formed with the input tangent vectors as columns.
The size of the output is `M x M` if `skipfirst` is false
and `M-1 x M-1` if it is true.
"""
function gs!(xs::Coupled{M}, skipfirst::Bool=true) where {M}
    N, Δ = skipfirst == true ? (M-1, 1) : (M, 0)
    R = UpperTriangular(zeros(N, N))
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
