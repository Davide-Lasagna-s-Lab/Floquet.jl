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
    pair_locations(B::AbstractMatrix, θtol::Real=1e-12)

Detect blocks like `[c -s; s c]` on the diagonal of matrix `B`, up to a
tolerance `θtol`, and return a vector of indices containing the `top-left`
locations of the blocks. Two-by-two blocks that look like the identity
matrix are not considered.
"""
function pair_locations(B::AbstractMatrix, θtol::Real=1e-12)
    N = size(B, 1)
    idxs = Int[]

    # a 1x1 matrix has no 2x2 block
    N < 2 && return idxs

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

"""
    floquet(L, N::Int, dt::Real, store::RAMStorage; kwargs...)

Compute the `N` leading Floquet exponents of the periodic orbit held in
`store`, with the matrix-free algorithm of Lasagna (2020), appendix A.

`N` tangent vectors, initially the first `N` canonical directions, are
marched over one period with the linearised equations along the stored
orbit, and re-orthonormalised by two-pass Gram-Schmidt every `oneevery`
steps, keeping the triangular factors. After each period, the orthogonal
matrix that best maps the final basis onto the initial one (a Procrustes
problem) separates real multipliers from complex-conjugate pairs. The
periods are repeated, starting each from the final basis of the previous
one, until the log-multipliers change by less than `λtol`.

`L` is the linearised operator, a callable `L(t, u, y, dydt)` that writes
in `dydt` the action on the tangent vector `y` of the Jacobian at the base
state `u`, interpolated from `store` at time `t`. `L` and `A` are called
for one tangent vector at a time, so they may own work buffers.

`store` holds one period of the orbit, for instance a `RAMStorage` built
with `period=T` and `storelast=false` and filled by integrating the orbit
over `(0, T)`. Such a storage must not hold a sample at or near `t = T`.
The period is `T = timespan(store)[2] - timespan(store)[1]`. A
non-periodic storage holding both endpoints works too, and is needed for
relative periodic orbits. `dt` is the time step of the tangent integration.

# Keyword arguments
  - `A = nothing`: linear part of the linearised operator, treated
    implicitly by the IMEX schemes. `L` is then the explicit part only.
  - `method = :RK4`: integration scheme of the tangent equations, the
    explicit `:RK4`, or one of the IMEX schemes `:CNRK2`, `:CB3R2R2`,
    `:CB3R2R3c`, `:CB3R2R3e` and `:CB4R3R4`, to be used with `A`.
  - `oneevery = 1`: number of time steps between re-orthonormalisations.
  - `shift = identity`: function applied in place to each tangent vector
    at the end of every period, before the Procrustes step. For a relative
    periodic orbit it applies the symmetry operation that maps the final
    state of the orbit back onto the initial state.
  - `init = nothing`: initial tangent vectors, `N` states of the type of the
    samples of `store`, orthonormalised before use. The default is the first
    `N` canonical directions, `y[i] = 1`. Pass `init` when these are not
    meaningful directions of the state, for instance for Fourier fields with
    padding or symmetry constraints.
  - `extract = :schur`: how the exponents follow from the factors of each
    period. `:schur` takes the diagonal of the triangular factors, and the
    2x2 blocks of complex pairs, as in the periodic Schur decomposition. It
    is accurate for all exponents, however far apart, but each one converges
    at the rate set by its gap to the next. `:ritz` takes the eigenvalues of
    the monodromy restricted to the subspace of the vectors, the Ritz values.
    They converge at the rate set by the first multiplier left out, also
    within clusters of nearly equal multipliers, but the exponent `λk` is
    computed with a relative error of about `1e-16 exp((λ1 - λk) T)` in its
    multiplier. Use `:ritz` for short orbits with clustered exponents.
  - `λtol = 1e-7`: tolerance on the change of the log-multipliers between
    two consecutive periods.
  - `θtol = 1e-7`: tolerance used to detect the 2x2 rotation blocks of the
    Procrustes matrix that flag complex-conjugate pairs.
  - `maxiter = 50`: maximum number of periods.
  - `verbose = false`: print the exponents after each period.

# Returns
A named tuple `(exponents, converged, periods, history)`. `exponents` is a
vector of `N` complex Floquet exponents, the log-multipliers divided by
`T`, sorted by decreasing real part. Imaginary parts are defined modulo
`2π/T` and lie in `[-π/T, π/T]`: a negative real multiplier has imaginary
part `π/T`, and the member of a complex pair with positive imaginary part
comes first. `converged` is `true` if the tolerance was met, `periods` is
the number of periods marched, and `history` holds the exponents after
each period.
"""
function floquet(L,
                 N::Int,
                 dt::Real,
                 store::RAMStorage;
                 A=nothing,
                 method::Symbol=:RK4,
                 oneevery::Int=1,
                 shift=identity,
                 init=nothing,
                 extract::Symbol=:schur,
                 λtol::Real=1e-7,
                 θtol::Real=1e-7,
                 maxiter::Int=50,
                 verbose::Bool=false)

    # ---- input checks ----
    # A periodic storage with a sample at (or just before) t = T duplicates
    # the sample at t = 0, and its interpolation is then wrong.
    ts = times(store)
    method in (:RK4, :CNRK2, :CB3R2R2, :CB3R2R3c, :CB3R2R3e, :CB4R3R4) ||
        throw(ArgumentError("method must be :RK4, :CNRK2, :CB3R2R2, " *
                            ":CB3R2R3c, :CB3R2R3e or :CB4R3R4"))
    (method !== :RK4 || A === nothing) ||
        throw(ArgumentError("method :RK4 is explicit, use an IMEX method with A"))
    extract in (:schur, :ritz) ||
        throw(ArgumentError("extract must be :schur or :ritz"))
    (init === nothing || length(init) == N) ||
        throw(ArgumentError("init must hold N = $N tangent vectors"))
    (!isperiodic(store) || period(store) - ts[end] > (ts[2] - ts[1])/2) ||
        throw(ArgumentError("periodic storage holds a sample at or near t = T, " *
                            "build it with storelast=false"))

    # ---- period of the base orbit ----
    span = timespan(store)
    T    = span[2] - span[1]

    # ---- initial tangent vectors: `init`, or the first N canonical directions ----
    x       = similar(samples(store)[1])
    qs_init = couplecopy(N, x)
    for i = 1:N
        if init === nothing
            qs_init[i] .= 0
            qs_init[i][i] = 1
        else
            qs_init[i] .= init[i]
        end
    end
    gs!(qs_init, false)
    qs_next = copy(qs_init)

    # ---- tangent flow ----
    # N linearised equations, each one seeing only its own tangent vector and
    # the base orbit interpolated from `store`. They share `L` and `A`, which
    # are called for one vector at a time. Copies would duplicate the FFT
    # plans that an operator may own, which must not be copied.
    scheme = getfield(Flows, method) # the Flows scheme named `method`
    ψ      = flow(couple(ntuple(i -> L, N)...),
                  couple(ntuple(i -> A, N)...),
                  CallDependency(ntuple(i -> (i,), N)...),
                  scheme(couplecopy(N, x), ContinuousMode(false)),
                  TimeStepFromStorage(dt))

    # ---- monitor ----
    # Gram-Schmidt on the tangent vectors at the start of each period, every
    # `oneevery` steps and at the end, storing the triangular factors R. The
    # template is a copy because the monitor evaluates the callback on it.
    mon = Monitor(copy(qs_next),
                  (t, qs) -> gs!(qs, false);
                  oneevery=oneevery)

    # ---- work arrays ----
    logλs     = ComplexF64[]               # log-multipliers of this period
    logλs_old = fill(ComplexF64(NaN), N)   # those of the previous period
    block     = zeros(2, 2)                # normalised product of 2x2 blocks
    P         = zeros(N, N)                # normalised product of the R factors
    C         = zeros(N, N)                # inner products of the initial and final vectors
    tmp       = zeros(2, 2)                # work array for the products
    history   = Vector{ComplexF64}[]       # exponents after each period

    for iter = 1:maxiter
        # ---- propagate the tangent vectors over one period ----
        ψ(qs_next, store, span, reset!(mon))

        # these are the R factors
        Rs = samples(mon)

        # ---- close relative periodic orbits ----
        for i = 1:N
            shift(qs_next[i])
        end

        # ---- Ritz values: eigenvalues of the monodromy restricted to the subspace ----
        # With F the monodromy, shift included, F Q₀ = Q₁ R, with R the product of
        # the factors, so that Q₀ᵀ F Q₀ = C R with C = Q₀ᵀ Q₁. The product is
        # normalised after each factor and its scale kept as a logarithm.
        if extract === :ritz
            P    .= Rs[1]
            scale = 0.0
            for p in 2:length(Rs)
                P    .= Rs[p] * P
                v     = maximum(abs, P)
                P   ./= v
                scale += log(v)
            end
            for i = 1:N, j = 1:N
                C[i, j] = dot(qs_init[i], qs_next[j])
            end
            append!(logλs, log.(complex.(eigvals(C * P))) .+ scale)
        end

        # ---- find best rotation matrix that takes qs_next to qs_init ----
        Q = procrustes(qs_init, qs_next, false)

        # find pairs of cc eigenvalues, the others are 1x1. With Ritz values
        # there is nothing left to do
        idxs_2x2 = extract === :ritz ? Int[] : pair_locations(Q, θtol)
        idxs_1x1 = extract === :ritz ? Int[] :
                   [i for i = 1:N if (i ∉ idxs_2x2) && (i-1 ∉ idxs_2x2)]

        # ---- eigenvalues of 2x2 blocks ----
        for i in idxs_2x2
            # multiply and normalise
            val = ComplexF64(0.0)
            @views block .= Rs[1][i:i+1, i:i+1]
            for p in 2:length(Rs)
                @views mul!(tmp, Rs[p][i:i+1, i:i+1], block)
                v = maximum(abs, tmp)
                block .= tmp/v
                val += log(ComplexF64(v))
            end

            # The Procrustes matrix gives qs_init ≈ qs_next*Q, so the monodromy
            # restricted to the basis qs_init is Qᵀ*R. The transpose only
            # changes the argument of the multipliers, not their modulus.
            @views mul!(tmp, transpose(Q[i:i+1, i:i+1]), block)

            # Before the basis has converged, the directions of two real
            # multipliers still rotate into each other and also give a 2x2
            # rotation block. Its eigenvalues are then real. Treat it as two
            # 1x1 blocks, and let the basis converge over the next periods.
            if (tmp[1, 1] - tmp[2, 2])^2 + 4*tmp[1, 2]*tmp[2, 1] ≥ 0
                append!(idxs_1x1, (i, i+1))
                continue
            end

            # calculate log-eigenvalues of the 2x2 block taking val into account
            vals = log.(eigvals(ComplexF64.(tmp))) .+ val

            # push cc eigenvalues with a consistent format, to help convergence checks
            _reλ = 0.5*(real(vals[1]) + real(vals[2]))
            _imλ = 0.5*(imag(vals[1]) - imag(vals[2]))
            push!(logλs, _reλ + im*abs(_imλ))
            push!(logλs, _reλ - im*abs(_imλ))
        end

        # ---- 1x1 blocks: when Q[i, i] = -1 (inverse hyperbolic) we add π ----
        for i in idxs_1x1
            logλ = sum(log(R[i, i]) for R in Rs)
            _im  = Q[i, i] < 0 ? Float64(π) : Float64(0)
            push!(logλs, logλ + im*_im)
        end

        # order by real part with unstable Floquet exponents first, and the
        # member of a pair with positive imaginary part before the other
        sort!(logλs; by=z -> (real(z), imag(z)), rev=true)

        # ---- store and print the exponents ----
        push!(history, logλs ./ T)
        if verbose
            @printf "period %3d:" iter
            foreach(λ -> @printf("  %+.10e%+.6ei", real(λ), imag(λ)), history[end])
            println()
            flush(stdout)
        end

        # ---- check convergence of the log-multipliers ----
        if maximum(abs, logλs .- logλs_old) < λtol
            return (exponents = history[end],
                    converged = true,
                      periods = iter,
                      history = history)
        end

        # ---- prepare the next period ----
        # store old exponents and update the basis vectors
        logλs_old .= logλs
        resize!(logλs, 0)
        qs_init .= qs_next
    end

    return (exponents = history[end],
            converged = false,
              periods = maxiter,
              history = history)
end
