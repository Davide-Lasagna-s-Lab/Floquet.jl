export floquet, pair_locations, tridiagdot!, procrustes

"""
    tridiagdot!(B::Tridiagonal, xs::Coupled{M}, ys::Coupled{M}, skipfirst::Bool) where {M}

Compute the dot product between elements `xs[i]` and elements `ys[j]` for `i` and 
`j` that satisfy `abs(i - j) ≤ 1`, arranging the values in a tridiagonal matrix.
If `skipfirst` is true, the dot product is only calculated for `i, j ≥ 1`.
"""
function tridiagdot!(B::Tridiagonal,
                    xs::Coupled{M},
                    ys::Coupled{M},
             skipfirst::Bool) where {M}
    N, Δ = skipfirst == true ? (M-1, 1) : (M, 0)
    size(B) == (N, N) || throw(ArgumentError("invalid size"))
    for i = 1:N;   B[i, i]   = dot(xs[i+Δ],   ys[i+Δ]);   end
    for i = 1:N-1; B[i, i+1] = dot(xs[i+Δ],   ys[i+Δ+1]); end
    for i = 1:N-1; B[i+1, i] = dot(xs[i+Δ+1], ys[i+Δ]);   end
    return B
end

"""
    procrustes(as::Coupled{M}, bs::Coupled{M}, skipfirst::Bool) where {M}

Return the orthogonal matrix `Q` that minimizes the frobenius norm of the 
rotated difference `A - BQ`, solving the so-called Procrustes problem. See
Golub and Van Loan, Matrix Computations, IV edition, pg 329.
"""
function procrustes(as::Coupled{M}, bs::Coupled{M}, skipfirst::Bool) where {M}
    N, Δ = skipfirst == true ? (M-1, 1) : (M, 0)
    fact = svd([dot(bs[i+Δ], as[j+Δ]) for i = 1:N, j = 1:N])
    return fact.U * fact.Vt
end

"""
    pair_locations(B::AbstractMatrix, tol::Real=1e-12)

Detect blocks like `[c -s; s c]` on the diagonal of matrix `B`, up to a 
tolerance `tol`, and return a vector of indices containing the `top-left`
locations of the blocks. Two-by-two blocks that look like the identity
matrix are not considered.
"""
function pair_locations(B::AbstractMatrix, θtol::Real=1e-12)
    N = size(B, 1)
    idxs = Int[]
    i = 1
    while true
        # look at 2x2 submatrix at position (i, i+1) and see if it looks
        # like a small rotation matrix of the form [c -s; s c]
        c1 = B[i, i]
        c2 = B[i+1, i+1]
        s1 = B[i, i+1]
        s2 = B[i+1, i]
        
        # we measure the angle induced by the two-by-two block on the vector
        # (1, 1) and check whether the rotation mod pi is larger than tol. We 
        # also check the entries c1, c2 and s1, s2 are similar enough, so we 
        # can consider the block a 2x2 rotation matrix
        θ = atan(c2 + s2, c1 + s1)
        if abs(mod(θ, π) - π/4) > θtol && (abs(c1 - c2) < θtol && abs(s1 + s2) < θtol)
            push!(idxs, i)
            i += 1
            i == N && break 
        end

        i += 1
        i == N && break
    end
    return idxs
end

function floquet(Lex, Lim,
                 N::Int, dt::Real,
                 store::AbstractStorage,
                 methodFun;
                 oneevery::Int=1,
                 tol::Real = 1e-7,
                 θtol::Real = 1e-7,
                 maxiter::Int=10,
                 verbose::Bool=true)
    # one element
    x = similar(samples(store)[1])

    # dependency list
    deps = calldeps(N, false);

    # and the integrator
    ψ = flow(opcouple(N, Lex),
             opcouple(N, Lim),
             deps,
             methodFun(couplecopy(N, x), ContinuousMode()),
             TimeStepFromStorage(dt))

    return _floquet(ψ, N, store, oneevery, λtol, θtol, maxiter, verbose)
end

function floquet(L,
                 N::Int, dt::Real,
                 store::AbstractStorage,
                 methodFun;
                 oneevery::Int=1,
                 λtol::Real = 1e-7,
                 θtol::Real = 1e-7,
                 maxiter::Int=10,
                 verbose::Bool=true)

    # one element
    x = similar(samples(store)[1])

    # dependency list
    deps = calldeps(N, false);

    # and the integrator
    ψ = flow(opcouple(N, L),
             deps,
             methodFun(couplecopy(N, x), ContinuousMode()),
             TimeStepFromStorage(dt))

    return _floquet(ψ, N, store, oneevery, λtol,  θtol, maxiter, verbose)
end

# actual routine
function _floquet(ψ,
                  N::Int,
                  store::AbstractStorage,
                  oneevery::Int=1,
                  λtol::Real = 1e-7,
                  θtol::Real = 1e-7,
                  maxiter::Int=10,
                  verbose::Bool=true)

    # this is the overall span
    span = timespan(store)

    # and its length
    T = span[2] - span[1]

    # one element
    x = similar(samples(store)[1])

    # initial conditions
    qs_init = couple(init_IC(N, x)...);
    qs_next = couple(init_IC(N, x)...);

    # define monitor
    mon = Monitor(qs_init,
                  qs->rfactor!(qs, false);
                  oneevery=oneevery);

    # preallocate
    logλs     = Vector{ComplexF64}(undef, 0)
    logλs_old = Vector{ComplexF64}(undef, N)

    # work arrays for the 2x2 matrix products
    block = zeros(2, 2)
    tmp   = zeros(2, 2)

    # fulltrace
    fulltrace = Vector{ComplexF64}[]

    for iter = 1:maxiter
        # propagate forward
        ψ(qs_next, store, span, reset!(mon));

        # these are the R factors
        Rs = samples(mon)

        # find best rotation matrix that takes qs_next to qs_init
        Q = procrustes(qs_init, qs_next, false)

        # find pairs of cc eigenvalues
        idxs_2x2 = pair_locations(Q, θtol)

        # the others are 1x1
        idxs_1x1 = [i for i = 1:N if (i ∉ idxs_2x2) && (i-1 ∉ idxs_2x2)]

        # find eigenvalues of 2x2 blocks
        for i in idxs_2x2
            # multiply and normalise
            val = ComplexF64(0.0)
            @views block .= Rs[1][i:i+1, i:i+1, 1]
            for p in 2:length(Rs)
                @views mul!(tmp, Rs[p][i:i+1, i:i+1], block)
                v = maximum(abs, tmp)
                block .= tmp/v
                val += log(ComplexF64(v))
            end
            @views mul!(tmp, Q[i:i+1, i:i+1], block)

            # calculate log-eigenvalues of the 2x2 block taking val into account
            vals = log.(eigvals(ComplexF64.(tmp))) .+ val

            #  push cc eigenvalues with a consistent format, to help convergence checks
            _reλ = 0.5*(real(vals[1]) + real(vals[2]))
            _imλ = 0.5*(imag(vals[1]) - imag(vals[2]))
            push!(logλs, _reλ + im*abs(_imλ))
            push!(logλs, _reλ - im*abs(_imλ))
        end

        # 1x1 blocks - when Q[i, i] = -1 (inverse hyperbolic) we add \pi
        for i in idxs_1x1
            logλ = sum(log(R[i, i]) for R in Rs)
            _im = Q[i, i] < 0 ? Float64(π) : Float64(0)
            push!(logλs, logλ + im*_im)
        end

        # order by real part with unstable Floquet exponents first
        sort!(logλs; by=real, rev=true)

        # store trace
        push!(fulltrace, copy(logλs))

        # debug
        if verbose == true
            println(logλs); flush(stdout)
        end

        # check convergence of the eigenvalues
        if maximum(abs, logλs .- logλs_old) < λtol
            return true, iter, logλs, fulltrace
        end

        # store old exponents for the next iteration
        logλs_old .= logλs
        resize!(logλs, 0)

        # update the basis vectors
        qs_init .= qs_next
    end

    return false, maxiter, logλs, fulltrace
end