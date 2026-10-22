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
    logmultipliers(Rs, Q, θtol)

Log-multipliers of one period, from the triangular factors `Rs` of the
re-orthonormalisations, in the order in which they were computed, and from
the orthogonal matrix `Q` that best maps the basis at the end of the period
onto the basis at its start (the Procrustes problem). Return a named tuple
`(logλs, index, pairs)`, with the log-multipliers sorted by decreasing real
part, the index in the basis of the vector of each of them, and the first
index of the two vectors of each complex-conjugate pair.

A real multiplier is the product of one diagonal entry of the factors, and
it is negative when the diagonal entry of `Q` is negative. A 2x2 rotation
block of `Q` flags a complex pair, whose multipliers are the eigenvalues of
the product of the 2x2 diagonal blocks of the factors, rotated back by the
block of `Q`. When these eigenvalues are real, the block is two real
multipliers whose vectors have not yet stopped rotating into each other.
"""
function logmultipliers(Rs, Q, θtol)
    N = size(Q, 1)

    # ---- pairs of complex multipliers, the others are 1x1 ----
    idxs_2x2 = pair_locations(Q, θtol)
    idxs_1x1 = [i for i = 1:N if (i ∉ idxs_2x2) && (i-1 ∉ idxs_2x2)]

    # ---- work arrays ----
    logλs = ComplexF64[]   # log-multipliers, in the order they are found
    index = Int[]          # index in the basis of the vector of each of them
    pairs = Int[]          # first index of each complex pair
    block = zeros(2, 2)    # normalised product of 2x2 blocks
    tmp   = zeros(2, 2)    # work array for the products

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

        # push cc eigenvalues with a consistent format, to help convergence
        # checks, the one with positive imaginary part on the first vector
        _reλ = 0.5*(real(vals[1]) + real(vals[2]))
        _imλ = 0.5*(imag(vals[1]) - imag(vals[2]))
        push!(logλs, _reλ + im*abs(_imλ))
        push!(logλs, _reλ - im*abs(_imλ))
        push!(index, i)
        push!(index, i+1)
        push!(pairs, i)
    end

    # ---- 1x1 blocks: when Q[i, i] = -1 (inverse hyperbolic) we add π ----
    for i in idxs_1x1
        logλ = sum(log(R[i, i]) for R in Rs)
        _im  = Q[i, i] < 0 ? Float64(π) : Float64(0)
        push!(logλs, logλ + im*_im)
        push!(index, i)
    end

    # ---- order by real part, unstable first, positive imaginary part first in a pair ----
    p = sortperm(logλs; by=z -> (real(z), imag(z)), rev=true)

    return (logλs = logλs[p],
            index = index[p],
            pairs = pairs)
end

"""
    covariant(U, V, pairs)

Floquet vectors and their adjoints at one time, from the forward and the
adjoint orthonormal bases `U` and `V`, two vectors of `N` states in basis
order, and from the first indices `pairs` of the two vectors of each complex
pair. The matrix of inner products `G[i, j] = dot(V[i], U[j])` is factored as
`G = L R` without pivoting, by blocks of one vector or of the two vectors of
a pair. Then `Φ = U R⁻¹` and `Ψ = V L⁻ᵀ`, so that `dot(Ψ[i], Φ[j]) = δᵢⱼ`.
Each block of `Φ` is made orthonormal, and `Ψ` is changed to keep the
biorthogonality. A real Floquet vector has unit norm, and the two vectors of
a pair are an orthonormal basis of its plane. Return `Φ` and `Ψ`.
"""
function covariant(U, V, pairs)
    N = length(U)

    # ---- blocks of one vector, or of the two vectors of a complex pair ----
    blocks = [i in pairs ? (i:i+1) : (i:i) for i = 1:N if i-1 ∉ pairs]

    # ---- block LU without pivoting of the inner products ----
    # No pivoting, because the order of the nested subspaces is the
    # information. A pair is one 2x2 pivot: the scalar pivot of its first
    # vector vanishes where the bases of the two planes are orthogonal,
    # the 2x2 pivot does not.
    G  = [dot(V[i], U[j]) for i = 1:N, j = 1:N]
    LG = Matrix{Float64}(I, N, N)
    RG = copy(G)
    for b in blocks
        rest = last(b)+1:N
        LG[rest, b]   = RG[rest, b] / RG[b, b]
        RG[rest, :] .-= LG[rest, b] * RG[b, :]
        RG[rest, b]  .= 0
    end

    # ---- Φ = U R⁻¹ and Ψ = V L⁻ᵀ, as combinations of states ----
    C = inv(RG)
    D = transpose(inv(LG))
    Φ = [similar(U[1]) for j = 1:N]
    Ψ = [similar(V[1]) for j = 1:N]
    for j = 1:N
        Φ[j] .= 0
        Ψ[j] .= 0
        for i = 1:N
            Φ[j] .+= C[i, j] .* U[i]
            Ψ[j] .+= D[i, j] .* V[i]
        end
    end

    # ---- orthonormal blocks of Φ, with Ψ changed to keep ⟨Ψ, Φ⟩ = I ----
    # Gram-Schmidt gives Φ_b = Q S, with S upper triangular, and leaves Q in
    # Φ_b. Then Ψ_b ← Ψ_b Sᵀ, in place from the first vector of the block.
    for b in blocks
        S = gs!(couple(Φ[b]...), false)
        for k in eachindex(b)
            Ψ[b[k]] .*= S[k, k]
            for m in k+1:length(b)
                Ψ[b[k]] .+= S[k, m] .* Ψ[b[m]]
            end
        end
    end

    return Φ, Ψ
end

"""
    floquet(L, N::Int, dt::Real, store::RAMStorage; kwargs...)

Compute the `N` leading Floquet exponents of the periodic or relative
periodic orbit held in `store`, with the matrix-free periodic
eigendecomposition of Lasagna (2020), appendix A, and optionally the Schur
vectors, the Floquet vectors and their adjoints along the orbit.

`N` tangent vectors, initially the first `N` canonical directions, are
marched over one period with the linearised equations along the stored
orbit, and re-orthonormalised by two-pass Gram-Schmidt every `oneevery`
steps, keeping the triangular factors. After each period, the orthogonal
matrix that best maps the final basis onto the initial one (a Procrustes
problem) separates real multipliers, positive or negative, from
complex-conjugate pairs. The periods are repeated, starting each from the
final basis of the previous one, until the log-multipliers change by less
than `λtol`. With `vectors = :floquet` the same iteration then runs
backwards in time with the adjoint linearised equations, and the Floquet
vectors and their adjoints follow at each time from an LU factorisation
without pivoting of the inner products of the forward and adjoint bases.

`L` is the linearised operator, a callable `L(t, u, y, dydt)` that writes
in `dydt` the action on the tangent vector `y` of the Jacobian at the base
state `u`, interpolated from `store` at time `t`. `L`, `A` and `Ladj` are
called for one vector at a time, so they may own work buffers.

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
    explicit `:RK4` or `:RK8`, or one of the IMEX schemes `:CNRK2`,
    `:CB3R2R2`, `:CB3R2R3c`, `:CB3R2R3e` and `:CB4R3R4`, to be used with
    `A`. `:RK8`, the eleven-stage, eighth-order scheme of Cooper and Verner
    (1972), needs Flows.jl with RK8, on the branch rk8 until it is merged.
  - `oneevery = 1`: number of time steps between re-orthonormalisations,
    also the stride of the returned times.
  - `shift = identity`: function applied in place to each tangent vector
    at the end of every period, before the Procrustes step. For a relative
    periodic orbit it applies the symmetry operation that maps the final
    state of the orbit back onto the initial state. It must preserve the
    inner product, since the vectors are not re-orthonormalised after it.
  - `init = nothing`: initial tangent vectors, `N` states of the type of the
    samples of `store`, orthonormalised before use. The default is the first
    `N` canonical directions, `y[i] = 1`. Pass `init` when these are not
    meaningful directions of the state, for instance for Fourier fields with
    padding or symmetry constraints.
  - `vectors = :none`: with `:schur`, also return the orthonormal (Schur)
    vectors at the times of the re-orthonormalisations of the last period,
    and the logarithms of the diagonals of the triangular factors. With
    `:floquet`, also run the adjoint sweep and return the adjoint bases,
    the Floquet vectors and their adjoints at the same times.
  - `Ladj = nothing`: adjoint linearised operator, needed by
    `vectors = :floquet`. It is a callable `Ladj(t, u, w, dwdt)`, like `L`,
    that writes in `dwdt` the adjoint of the Jacobian at `u` in the inner
    product `dot`, not its negative, applied to `w`. The adjoint equations
    `-dw/dt = J(u)ᵀ w` are marched backwards in time. With an IMEX scheme,
    `A` is also the implicit part of the adjoint equations, and must be
    self-adjoint, and `Ladj` is the adjoint of the explicit part `L`.
  - `shiftadj = identity`: adjoint of `shift`, its inverse, applied in place
    to each adjoint vector at the end of every backward period. It is needed
    by `vectors = :floquet` with a `shift`. For a reflection it is `shift`.
  - `λtol = 1e-7`: tolerance on the change of the log-multipliers between
    two consecutive periods.
  - `θtol = 1e-7`: tolerance used to detect the 2x2 rotation blocks of the
    Procrustes matrix that flag complex-conjugate pairs.
  - `maxiter = 50`: maximum number of periods, of each sweep.
  - `verbose = false`: print the exponents after each period, and those of
    the adjoint sweep.

# Returns
A named tuple with the following fields. Those that were not requested
with `vectors` are `nothing`.
  - `exponents`: `N` complex Floquet exponents, the log-multipliers divided
    by `T`, sorted by decreasing real part. Imaginary parts are defined
    modulo `2π/T` and lie in `[-π/T, π/T]`: a negative real multiplier has
    imaginary part `π/T`, and the member of a complex pair with positive
    imaginary part comes first.
  - `converged`: `true` if the tolerance was met, by both sweeps with
    `vectors = :floquet`.
  - `periods`: number of periods marched by the forward sweep.
  - `history`: exponents after each period of the forward sweep.
  - `order`: `order[i]` is the index, in the basis, of the vector of
    `exponents[i]`. It is `1:N` in generic cases.
  - `pairs`: first index, in the basis, of the two vectors of each
    complex-conjugate pair.
  - `times`: the `K + 1` times of the re-orthonormalisations of the last
    period, from the start of the period to its end.
  - `bases`: `bases[k][j]` is the `j`-th orthonormal vector at `times[k]`.
    The first `j` vectors span the invariant subspace of the `j` leading
    multipliers. At the end of the period they are taken before `shift`.
  - `stretching`: `N × (K + 1)` matrix, `stretching[j, k]` is the logarithm
    of the `j`-th diagonal entry of the triangular factor at `times[k]`, the
    stretching of vector `j` over the segment that ends there. The sum of
    row `order[i]` is `T` times the real part of `exponents[i]`, unless the
    exponent belongs to a pair, for which only the sum of its two rows has a
    meaning.
  - `adjointexponents`: exponents of the adjoint sweep. They equal
    `exponents` up to the error of the integration and the tolerance.
  - `adjointbases`: `adjointbases[k][j]` is the `j`-th orthonormal vector
    of the adjoint sweep at `times[k]`. The first `j` vectors span the
    invariant subspace of the `j` leading multipliers of the adjoint.
  - `floquetvectors`: `floquetvectors[k][j]` is the `j`-th Floquet vector
    at `times[k]`, of unit norm. For a pair, the two vectors are an
    orthonormal basis of its plane.
  - `adjointvectors`: `adjointvectors[k][j]` is the `j`-th adjoint Floquet
    vector at `times[k]`, with `dot(adjointvectors[k][i],
    floquetvectors[k][j]) = δᵢⱼ`.

All the vectors and the rows of `stretching` are in basis order, so that
the Floquet vector of `exponents[i]` is `floquetvectors[k][order[i]]`. The
Floquet vectors at the end of the period are those at its start moved by
the inverse of `shift` and multiplied by the sign of their multiplier, and
the basis of the plane of a pair is rotated. Multipliers of equal modulus
give a plane and not individual vectors. `vectors = :schur` keeps
`(K + 1) N` states and `vectors = :floquet` keeps `4 (K + 1) N`.
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
                 vectors::Symbol=:none,
                 Ladj=nothing,
                 shiftadj=identity,
                 λtol::Real=1e-7,
                 θtol::Real=1e-7,
                 maxiter::Int=50,
                 verbose::Bool=false)

    # ---- input checks ----
    # A periodic storage with a sample at (or just before) t = T duplicates
    # the sample at t = 0, and its interpolation is then wrong.
    ts = times(store)
    method in (:RK4, :RK8, :CNRK2, :CB3R2R2, :CB3R2R3c, :CB3R2R3e, :CB4R3R4) ||
        throw(ArgumentError("method must be :RK4, :RK8, :CNRK2, :CB3R2R2, " *
                            ":CB3R2R3c, :CB3R2R3e or :CB4R3R4"))
    isdefined(Flows, method) ||
        throw(ArgumentError("method :$method is not defined in this version of Flows.jl, " *
                            ":RK8 needs Flows.jl with RK8 (branch rk8 until it is merged)"))
    (method ∉ (:RK4, :RK8) || A === nothing) ||
        throw(ArgumentError("methods :RK4 and :RK8 are explicit, use an IMEX method with A"))
    vectors in (:none, :schur, :floquet) ||
        throw(ArgumentError("vectors must be :none, :schur or :floquet"))
    (vectors !== :floquet || Ladj !== nothing) ||
        throw(ArgumentError("vectors = :floquet needs the adjoint operator Ladj"))
    (vectors !== :floquet || shift === identity || shiftadj !== identity) ||
        throw(ArgumentError("vectors = :floquet with a shift needs its adjoint shiftadj"))
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
    # `oneevery` steps and at the end, storing the triangular factors R. With
    # `vectors` other than :none, the orthonormal vectors are also copied in
    # `bases`, overwritten at each period, so that it ends with those of the
    # last one. The template is a copy because the monitor evaluates the
    # callback on it, twice, which only adds two buffers.
    bases = Vector{typeof(x)}[]   # orthonormal vectors at the times of a period
    count = Ref(0)                # re-orthonormalisations so far in the period
    function observe(t, qs)
        R = gs!(qs, false)
        if vectors !== :none
            count[] += 1
            count[] > length(bases) && push!(bases, [similar(x) for j = 1:N])
            for j = 1:N
                bases[count[]][j] .= qs[j]
            end
        end
        return R
    end
    mon = Monitor(copy(qs_next), observe; oneevery=oneevery)

    # ---- work arrays ----
    logλs_old = fill(ComplexF64(NaN), N)   # log-multipliers of the previous period
    history   = Vector{ComplexF64}[]       # exponents after each period
    order     = collect(1:N)               # index in the basis of the vector of each exponent
    pairs     = Int[]                      # first index in the basis of each complex pair
    periods   = 0                          # number of periods marched
    converged = false                      # whether the tolerance was met

    for iter = 1:maxiter
        # ---- propagate the tangent vectors over one period ----
        count[] = 0
        ψ(qs_next, store, span, reset!(mon))

        # ---- close relative periodic orbits ----
        for i = 1:N
            shift(qs_next[i])
        end

        # ---- log-multipliers, from the R factors and the rotation between the bases ----
        Q       = procrustes(qs_init, qs_next, false)
        lm      = logmultipliers(samples(mon), Q, θtol)
        order   = lm.index
        pairs   = lm.pairs
        periods = iter

        # ---- store and print the exponents ----
        push!(history, lm.logλs ./ T)
        if verbose
            @printf "period %3d:" iter
            foreach(λ -> @printf("  %+.10e%+.6ei", real(λ), imag(λ)), history[end])
            println()
            flush(stdout)
        end

        # ---- check convergence of the log-multipliers ----
        if maximum(abs, lm.logλs .- logλs_old) < λtol
            converged = true
            break
        end

        # ---- prepare the next period ----
        # store old exponents and update the basis vectors
        logλs_old .= lm.logλs
        qs_init   .= qs_next
    end

    # ---- Schur vectors and stretching along the last period ----
    # The calls of the monitor on its template may have left extra buffers.
    tmon       = nothing
    stretching = nothing
    if vectors !== :none
        resize!(bases, count[])
        tmon       = copy(times(mon))
        stretching = reduce(hcat, [log.(diag(R)) for R in samples(mon)])
    end

    # ---- adjoint sweep, Floquet vectors and adjoints ----
    adjointexponents = nothing
    adjointbases     = nothing
    floquetvectors   = nothing
    adjointvectors   = nothing
    if vectors === :floquet
        # ---- adjoint flow ----
        # The scheme in adjoint mode, over a decreasing span, integrates
        # -dw/dt = Ladj(t, u, w), with A as its implicit part, which must be
        # self-adjoint. Ladj is the adjoint of the Jacobian, not its negative.
        φ = flow(couple(ntuple(i -> Ladj, N)...),
                 couple(ntuple(i -> A, N)...),
                 CallDependency(ntuple(i -> (i,), N)...),
                 scheme(couplecopy(N, x), ContinuousMode(true)),
                 TimeStepFromStorage(dt))

        # ---- initial adjoint vectors: the forward basis at T, before the shift ----
        # It gives the adjoint basis the order of the forward one, which the
        # LU needs, and it satisfies the constraints of the state.
        K  = length(tmon) - 1
        ws = couplecopy(N, x)
        for j = 1:N
            ws[j] .= bases[K+1][j]
        end
        ws_init = copy(ws)

        # ---- work arrays ----
        adjointbases = [[similar(x) for j = 1:N] for k = 1:K+1]  # adjoint bases at the times
        Rs           = eltype(samples(mon))[]                    # factors of one backward period
        logμs        = fill(ComplexF64(NaN), N)                  # adjoint log-multipliers
        logμs_old    = fill(ComplexF64(NaN), N)                  # those of the previous period
        adjconverged = false                                     # whether the tolerance was met

        for iter = 1:maxiter
            # ---- march back over one period, segment by segment ----
            # Going from each forward time to the previous one puts the
            # adjoint bases at the same times, whatever `oneevery` is.
            empty!(Rs)
            for j = 1:N
                adjointbases[K+1][j] .= ws[j]
            end
            for k = K+1:-1:2
                φ(ws, store, (tmon[k], tmon[k-1]))
                push!(Rs, gs!(ws, false))
                for j = 1:N
                    adjointbases[k-1][j] .= ws[j]
                end
            end

            # ---- move the adjoint vectors to the frame of the end of the period ----
            for j = 1:N
                shiftadj(ws[j])
            end

            # ---- log-multipliers, with the extraction of the forward sweep ----
            # A rotation block with real eigenvalues is two real multipliers
            # whose vectors still rotate, and must not stop the iteration.
            Q     = procrustes(ws_init, ws, false)
            logμs = logmultipliers(Rs, Q, θtol).logλs

            # ---- print the adjoint exponents ----
            if verbose
                @printf "adjoint %3d:" iter
                foreach(λ -> @printf("  %+.10e%+.6ei", real(λ), imag(λ)), logμs ./ T)
                println()
                flush(stdout)
            end

            # ---- check convergence of the adjoint log-multipliers ----
            if maximum(abs, logμs .- logμs_old) < λtol
                adjconverged = true
                break
            end

            # ---- prepare the next backward period ----
            logμs_old .= logμs
            ws_init   .= ws
        end

        # ---- Floquet vectors and adjoints at each time ----
        # The pairs are those of the forward sweep, in basis order.
        floquetvectors = similar(adjointbases)
        adjointvectors = similar(adjointbases)
        for k = 1:K+1
            floquetvectors[k], adjointvectors[k] = covariant(bases[k], adjointbases[k], pairs)
        end

        converged        = converged && adjconverged
        adjointexponents = logμs ./ T
    end

    # ---- assemble ----
    return (exponents = history[end],
            converged = converged,
              periods = periods,
              history = history,
                order = order,
                pairs = pairs,
                times = tmon,
                bases = vectors === :none ? nothing : bases,
           stretching = stretching,
     adjointexponents = adjointexponents,
         adjointbases = adjointbases,
       floquetvectors = floquetvectors,
       adjointvectors = adjointvectors)
end
