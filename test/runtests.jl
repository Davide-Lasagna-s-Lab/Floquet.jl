using LinearAlgebra
using Random
using Flows
using Floquet
using Test

Random.seed!(0)

# ---------------------------------------------------------------------------- #
# HELPERS

# Fill a periodic storage with one period of the orbit of `f` through `x0`,
# using RK4 steps of size T/n. When the rounded time grid is not made of
# exactly n steps, a sample falls just before t = T: then try n + 1.
function periodic_storage(f, x0, T, n)
    while true
        store = RAMStorage(zero(x0); period=T, storelast=false, degree=5)
        F     = flow(f, RK4(zero(x0)), TimeStepConstant(T/n))
        F(copy(x0), (0, T), store)
        length(times(store)) == n && return store
        n += 1
    end
end

# Storage holding the equilibrium x = 0 of a linear system in n dimensions,
# seen as a periodic orbit of arbitrary period T.
zero_orbit(n, T) = periodic_storage((t, x, dxdt) -> (dxdt .= 0; dxdt), zeros(n), T, 400)

# Distance between two sets of Floquet exponents, comparing the imaginary
# parts modulo 2π/T through the phase of the multipliers. Each exponent is
# matched to the closest one of the other set, in both directions.
function distance(λs, μs, T)
    d(λ, μ) = abs(real(λ) - real(μ)) + abs(cis(imag(λ)*T) - cis(imag(μ)*T))/T
    return max(maximum(minimum(d(λ, μ) for μ in μs) for λ in λs),
               maximum(minimum(d(λ, μ) for λ in λs) for μ in μs))
end

# Distance of the vector v from the span of the columns of B, relative to |v|
function offspan(v, B)
    Q = Matrix(qr(float.(hcat(B))).Q)
    return norm(v - Q*(Q'*v))/norm(v)
end

# Largest deviation from the identity of the inner products ⟨ψᵢ, φⱼ⟩
biorth(Ψ, Φ) = maximum(abs, [dot(ψ, φ) for ψ in Ψ, φ in Φ] - I)

# Integrate one tangent vector y over `span` along the orbit in `store`, with
# the linearised equations, or with the adjoint ones over a decreasing span
function propagate(L, dt, store, y, span; A=nothing, method=:RK4, backward=false)
    scheme = getfield(Flows, method)(similar(y), ContinuousMode(backward))
    F      = A === nothing ? flow(L, scheme, TimeStepFromStorage(dt)) :
                             flow(L, A, scheme, TimeStepFromStorage(dt))
    return F(copy(y), store, span)
end

# Lorenz equations and their linearisation, L(t, u, y, dydt) as used with
# a storage, and (t, x, dxdt, y, dydt) as used in a coupled integration
function lorenz(t, x, dxdt)
    dxdt[1] = 10*(x[2] - x[1])
    dxdt[2] = 28*x[1] - x[2] - x[1]*x[3]
    dxdt[3] = -8/3*x[3] + x[1]*x[2]
    return dxdt
end

function lorenz_tan(t, u, y, dydt)
    dydt[1] = 10*(y[2] - y[1])
    dydt[2] = (28 - u[3])*y[1] - y[2] - u[1]*y[3]
    dydt[3] = u[2]*y[1] + u[1]*y[2] - 8/3*y[3]
    return dydt
end

lorenz_tan(t, x, dxdt, y, dydt) = lorenz_tan(t, x, y, dydt)

# explicit part of the linearisation, for IMEX schemes with the diagonal
# Diagonal([-10, -1, -8/3]) of the Jacobian treated implicitly
function lorenz_tan_ex(t, u, y, dydt)
    dydt[1] = 10*y[2]
    dydt[2] = (28 - u[3])*y[1] - u[1]*y[3]
    dydt[3] = u[2]*y[1] + u[1]*y[2]
    return dydt
end

# adjoint of the linearisation, the transpose of the Jacobian applied to w,
# and the adjoint of its explicit part
function lorenz_adj(t, u, w, dwdt)
    dwdt[1] = -10*w[1] + (28 - u[3])*w[2] + u[2]*w[3]
    dwdt[2] = 10*w[1] - w[2] + u[1]*w[3]
    dwdt[3] = -u[1]*w[2] - 8/3*w[3]
    return dwdt
end

function lorenz_adj_ex(t, u, w, dwdt)
    dwdt[1] = (28 - u[3])*w[2] + u[2]*w[3]
    dwdt[2] = 10*w[1] + u[1]*w[3]
    dwdt[3] = -u[1]*w[2]
    return dwdt
end

# Final state and monodromy matrix of the Lorenz orbit through x0 over a
# time T, by RK4 with n steps on the orbit and three tangent vectors
function monodromy(x0, T, n)
    G = flow(couple(lorenz, lorenz_tan, lorenz_tan, lorenz_tan),
             CallDependency((1,), (1, 2), (1, 3), (1, 4)),
             RK4(couplecopy(4, zeros(3))),
             TimeStepConstant(T/n))
    z = couple(copy(x0), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    G(z, (0, T))
    return z[1], hcat(z[2], z[3], z[4])
end

# Newton shooting for a Lorenz periodic orbit, with x0[3] fixed on the
# Poincaré section: the unknowns are x0[1], x0[2] and the period T
function shoot(x0, T, n)
    x = copy(x0)
    for iter = 1:20
        xT, M = monodromy(x, T, n)
        r     = xT .- x
        norm(r) < 1e-9 && return x, T
        J     = hcat((M - I)[:, 1:2], lorenz(0.0, xT, zeros(3)))
        δ     = J \ r
        x[1] -= δ[1]
        x[2] -= δ[2]
        T    -= δ[3]
    end
    error("Newton shooting did not converge")
end

# A point of the Lorenz orbit AB on the plane z = 27, and its period,
# converged by Newton shooting with 4000 RK4 steps, as in the README
const x0_AB = [-13.76361068212564, -19.57875194244233, 27.0]
const T_AB  = 1.558652210720224

# The Lorenz equations coupled to a complex amplitude a = p + iq, which obeys
# a' = (μ + iνz) a - |a|² a, here with μ = ν = 1. The system is invariant
# under the rotations a → exp(iθ) a. A periodic orbit of the Lorenz equations
# with |a|² = μ is a relative periodic orbit, which rotates a by ν times the
# integral of z over the period. Its exponents are those of the Lorenz orbit,
# 0 for the rotation and -2μ for the radius |a|.
function lorenzphase(t, u, dudt)
    x, y, z, p, q = u
    r2 = p^2 + q^2
    dudt[1] = 10*(y - x)
    dudt[2] = 28*x - y - x*z
    dudt[3] = -8/3*z + x*y
    dudt[4] = p - z*q - r2*p
    dudt[5] = q + z*p - r2*q
    return dudt
end

function lorenzphase_jac(u)
    x, y, z, p, q = u
    return [-10.0  10.0  0.0   0.0             0.0;
            28-z   -1.0  -x    0.0             0.0;
            y      x     -8/3  0.0             0.0;
            0.0    0.0   -q    1 - 3p^2 - q^2  -z - 2p*q;
            0.0    0.0   p     z - 2p*q        1 - p^2 - 3q^2]
end

lorenzphase_tan(t, u, y, dydt) = mul!(dydt, lorenzphase_jac(u), y)
lorenzphase_adj(t, u, w, dwdt) = mul!(dwdt, transpose(lorenzphase_jac(u)), w)


# ---------------------------------------------------------------------------- #
# BUILDING BLOCKS

@testset "procrustes                             " begin
    N = 5
    B = rand(2*N, N)
    Q_exact = Matrix(qr(rand(N, N)).Q)
    A = B*Q_exact

    as = couple([A[:, i] for i = 1:N]...)
    bs = couple([B[:, i] for i = 1:N]...)

    Q = Floquet.procrustes(as, bs, false)
    @test norm(Q_exact .- Q) < 1e-14
    @test norm(A - B*Q) < 1e-14
end

@testset "pair_locations                         " begin
    N = 5
    B = zeros(N, N) + I
    @test Floquet.pair_locations(B, 1e-6) == Int[]
    @test Floquet.pair_locations(ones(1, 1), 1e-6) == Int[]

    θ = π/4; c, s = cos(θ), sin(θ)
    B = zeros(N, N) + I
    B[2, 2] = c; B[3, 3] = c; B[2, 3] = s; B[3, 2] = -s
    @test Floquet.pair_locations(B, 1e-6) == Int[2]

    B = zeros(N, N) + I
    B[2, 2] = -c; B[3, 3] = c; B[2, 3] = s; B[3, 2] = -s
    @test Floquet.pair_locations(B, 1e-6) == Int[]

    B = [0.9999999999994441  1.0546919551626985e-6;
        -1.05469195515357e-6 0.9999999999989968]
    @test Floquet.pair_locations(B, 1e-6) == Int[1]
    @test Floquet.pair_locations(B, 1e-5) == Int[]

    B = [0.9999999999999919    1.2968516247787576e-7;
        -1.2968516246475442e-7 0.999999999999988]
    @test Floquet.pair_locations(B, 1e-7) == Int[1]
    @test Floquet.pair_locations(B, 1e-6) == Int[]
    @test Floquet.pair_locations(B, 1e-5) == Int[]

    B = [1    -1.1e-4;
         1.1e-4  1]
    @test Floquet.pair_locations(B,   1e-4) == Int[1]
    @test Floquet.pair_locations(B, 1.1e-4) == Int[]
end

@testset "gs!                                    " begin
    for N = (3, 5, 10)
        A1 = randn(N, N)
        A2 = copy(A1)
        f  = qr(A1)
        r1 = Floquet.gs!(couple(nothing, [view(A1, :, i) for i = 1:N]...), true)
        r2 = Floquet.gs!(couple([view(A2, :, i) for i = 1:N]...), false)

        # check diagonal
        for i = 1:N
            @test abs(abs(r1[i, i]) - abs(f.R[i, i])) < 1e-13
            @test abs(abs(r2[i, i]) - abs(f.R[i, i])) < 1e-13
        end

        # check orthogonality of vectors
        @test maximum(abs, A1'*A1 - I) < 1e-13
        @test maximum(abs, A2'*A2 - I) < 1e-13
    end
end

@testset "logmultipliers                         " begin
    # two triangular factors, and a Procrustes rotation with a real
    # multiplier, a complex pair at (2, 3) and a negative multiplier. The
    # monodromy restricted to the basis is Qᵀ R₂ R₁, block upper triangular,
    # and its log-eigenvalues are the log-multipliers.
    R1 = [2.0 0.3 -0.2  0.5;
          0.0 1.1  0.2  0.1;
          0.0 0.0  0.9 -0.4;
          0.0 0.0  0.0  3.0]
    R2 = [1.5 -0.1  0.4 0.2;
          0.0  1.0 -0.1 0.3;
          0.0  0.0  1.2 0.6;
          0.0  0.0  0.0 2.0]
    c  = cos(0.7)
    s  = sin(0.7)
    Q  = [1.0 0.0 0.0  0.0;
          0.0 c   -s   0.0;
          0.0 s   c    0.0;
          0.0 0.0 0.0 -1.0]
    lm = Floquet.logmultipliers([R1, R2], Q, 1e-7)
    @test issorted(real.(lm.logλs); rev=true)
    @test distance(lm.logλs, log.(complex.(eigvals(transpose(Q)*R2*R1))), 1.0) < 1e-12
    @test imag(lm.logλs[1]) ≈ π
    @test lm.index == [4, 1, 2, 3]
    @test lm.pairs == [2]

    # a rotation block with real eigenvalues is two real multipliers whose
    # vectors still rotate into each other, not a pair
    c  = cos(0.1)
    s  = sin(0.1)
    Q  = [1.0 0.0 0.0 0.0;
          0.0 c   -s  0.0;
          0.0 s   c   0.0;
          0.0 0.0 0.0 1.0]
    lm = Floquet.logmultipliers([Matrix(Diagonal([1.0, 4.0, 0.25, 0.5]))], Q, 1e-7)
    @test lm.pairs == Int[]
    @test lm.logλs ≈ log.([4.0, 1.0, 0.5, 0.25])
    @test lm.index == [2, 1, 4, 3]
end

@testset "covariant                              " begin
    # Floquet vectors Φ and adjoints Ψ = Φ⁻ᵀ, and the nested orthonormal
    # bases of the forward and adjoint subspace iterations, from the QR
    # factorisations of Φ and Ψ
    rng = MersenneTwister(2)
    n   = 6
    Φt  = randn(rng, n, n)
    Ψt  = Matrix(transpose(inv(Φt)))
    U   = [Matrix(qr(Φt).Q)[:, j] for j = 1:n]
    V   = [Matrix(qr(Ψt).Q)[:, j] for j = 1:n]

    # real multipliers: each vector along its column, of unit norm
    Φ, Ψ = Floquet.covariant(U, V, Int[])
    @test maximum(j -> offspan(Φ[j], Φt[:, j]), 1:n) < 1e-12
    @test maximum(j -> offspan(Ψ[j], Ψt[:, j]), 1:n) < 1e-12
    @test maximum(j -> abs(norm(Φ[j]) - 1), 1:n) < 1e-14
    @test biorth(Ψ, Φ) < 1e-12

    # a complex pair at (2, 3): an orthonormal basis of the plane of its columns
    Φ, Ψ = Floquet.covariant(U, V, [2])
    @test maximum(j -> offspan(Φ[j], Φt[:, 2:3]), 2:3) < 1e-12
    @test maximum(j -> offspan(Ψ[j], Ψt[:, 2:3]), 2:3) < 1e-12
    @test maximum(j -> offspan(Φ[j], Φt[:, j]), (1, 4, 5, 6)) < 1e-12
    @test abs(dot(Φ[2], Φ[3])) < 1e-14
    @test maximum(j -> abs(norm(Φ[j]) - 1), 1:n) < 1e-14
    @test biorth(Ψ, Φ) < 1e-12

    # the basis of the plane is the one closest to the vectors of U of the
    # pair, for which ⟨φᵢ, uⱼ⟩ is symmetric and positive definite on the pair
    B = [dot(Φ[i], U[j]) for i = 2:3, j = 2:3]
    @test norm(B - B') < 1e-12
    @test minimum(eigvals(Symmetric(B))) > 0

    # a leading pair at (1, 2), whose plane is that of the first two vectors
    # of U: the basis is those two vectors
    Φ, Ψ = Floquet.covariant(U, V, [1])
    @test norm(Φ[1] - U[1]) + norm(Φ[2] - U[2]) < 1e-12
    @test biorth(Ψ, Φ) < 1e-12
end

@testset "input checks                           " begin
    store = zero_orbit(2, 1.0)
    L(t, u, y, dydt) = (dydt .= y; dydt)
    @test_throws ArgumentError floquet(L, 2, 1e-2, store; method=:Euler)
    @test_throws ArgumentError floquet(L, 2, 1e-2, store; A=Diagonal([1.0, 1.0]))

    # a periodic storage holding the endpoint t = T
    bad = RAMStorage(zeros(2); period=1.0)
    F   = flow((t, x, dxdt) -> (dxdt .= 0; dxdt), RK4(zeros(2)), TimeStepConstant(0.01))
    F(zeros(2), (0, 1), bad)
    @test_throws ArgumentError floquet(L, 2, 1e-2, bad)

    # the vectors, and the adjoint operator and shift they need
    flip(q) = (q .*= -1; q)
    @test_throws ArgumentError floquet(L, 2, 1e-2, store; vectors=:all)
    @test_throws ArgumentError floquet(L, 2, 1e-2, store; vectors=:floquet)
    @test_throws ArgumentError floquet(L, 2, 1e-2, store; vectors=:floquet, Ladj=L, shift=flip)

    # the keyword extract was removed
    @test_throws MethodError floquet(L, 2, 1e-2, store; extract=:schur)
end


# ---------------------------------------------------------------------------- #
# LINEAR SYSTEMS WITH KNOWN EXPONENTS

@testset "constant coefficients                  " begin
    # x' = A x with eigenvalues 0.5, 0.1 ± 2i, -0.2 and -1.3, in a random basis
    n = 5
    D = zeros(n, n)
    D[1:2, 1:2] = [0.1 -2.0; 2.0 0.1]
    D[3, 3] = 0.5
    D[4, 4] = -0.2
    D[5, 5] = -1.3
    V = randn(n, n)
    A = V*D/V
    μs = eigvals(A)

    # the equilibrium x = 0 has exponents eig(A) for any period T, with the
    # imaginary parts defined modulo 2π/T. With these numbers of steps, the
    # rounded time grid used to end one ulp past T, before Flows 5e07a90.
    L(t, u, y, dydt) = mul!(dydt, A, y)
    for (T, nsteps) in ((3.7, 3000), (2.9, 3005))
        out = floquet(L, n, T/nsteps, zero_orbit(n, T); λtol=1e-11, maxiter=100)
        λs  = out.exponents
        @test out.converged
        @test issorted(real.(λs); rev=true)
        @test maximum(abs, sort(real.(λs)) - sort(real.(μs))) < 1e-8
        @test distance(λs, μs, T) < 1e-8
    end

    # the same with IMEX schemes, the diagonal of A being implicit, with a
    # tolerance set by the order of the scheme
    T   = 3.7
    Aim = Diagonal(diag(A))
    Aex = A - Aim
    L_ex(t, u, y, dydt) = mul!(dydt, Aex, y)
    for (method, tol) in ((:CB4R3R4,  1e-8),
                          (:CB3R2R3e, 1e-7),
                          (:CB3R2R3c, 1e-7),
                          (:CB3R2R2,  1e-4),
                          (:CNRK2,    1e-4))
        out = floquet(L_ex, n, 1e-3, zero_orbit(n, T); A=Aim, method=method,
                      λtol=1e-11, maxiter=100)
        @test out.converged
        @test distance(out.exponents, μs, T) < tol
    end

    # only the leading exponents, without splitting the complex pair
    μs_sorted = sort(μs; by=real, rev=true)
    for N in (1, 3)
        out = floquet(L, N, 1e-3, zero_orbit(n, T); λtol=1e-11, maxiter=100)
        @test out.converged
        @test distance(out.exponents, μs_sorted[1:N], T) < 1e-8
    end

    # the same from random initial vectors, not orthonormal
    for N in (1, 3)
        init = [randn(n) for i = 1:N]
        out  = floquet(L, N, 1e-3, zero_orbit(n, T); init=init, λtol=1e-11, maxiter=100)
        @test out.converged
        @test distance(out.exponents, μs_sorted[1:N], T) < 1e-8
    end
    @test_throws ArgumentError floquet(L, 3, 1e-3, zero_orbit(n, T); init=[randn(n)])
end

@testset "time-periodic coefficients             " begin
    # x' = R(t) A R(t)ᵀ x, with R(t) = exp(Ω t) a rotation in the (x1, x2)
    # plane at angular speed ω. In the rotating frame y = R(t)ᵀ x, we have
    # x' = R y' + Ω R y = R A y, so that y' = (A - Ω) y, using RᵀΩR = Ω. The
    # monodromy is then M = R(T) exp((A - Ω) T), and the exponents are
    # log(eig(M))/T. The time dependence enters through `t`, along the
    # equilibrium x = 0 seen as a periodic orbit of period T.
    n = 4
    rotation(ω, t) = [cos(ω*t) -sin(ω*t) 0 0;
                      sin(ω*t)  cos(ω*t) 0 0;
                      0         0        1 0;
                      0         0        0 1]

    # (a) ω T = 2π, so R(T) = I and M = exp((A - Ω) T). A - Ω is a random
    # matrix with eigenvalues 0.4, -0.1 ± 0.8i and -0.9.
    T = 4.0
    ω = 2π/T
    Ω = [0 -ω 0 0; ω 0 0 0; 0 0 0 0; 0 0 0 0]
    D = [0.4 0 0 0; 0 -0.1 -0.8 0; 0 0.8 -0.1 0; 0 0 0 -0.9]
    V = randn(n, n)
    A = V*D/V + Ω

    L(t, u, y, dydt) = mul!(dydt, rotation(ω, t)*A*rotation(ω, t)', y)
    out = floquet(L, n, 1e-3, zero_orbit(n, T); λtol=1e-11, maxiter=100)
    M   = exp(Ω*T)*exp((A - Ω)*T)
    μs  = log.(complex.(eigvals(M)))./T
    @test out.converged
    @test distance(out.exponents, μs, T) < 1e-8
    @test distance(out.exponents, eigvals(A - Ω), T) < 1e-8

    # (b) ω T = π, so R(T) = diag(-1, -1, 1, 1), and A is block-diagonal
    # for the coefficients to be T-periodic. A - Ω has real eigenvalues 0.3
    # and -0.5 in the rotating plane: the multipliers there are negative,
    # with exponents 0.3 + iπ/T and -0.5 + iπ/T. The other block has
    # eigenvalues -0.2 ± 0.6i.
    T = 2.0
    ω = π/T
    Ω = [0 -ω 0 0; ω 0 0 0; 0 0 0 0; 0 0 0 0]
    W = randn(2, 2)
    A = zeros(n, n)
    A[1:2, 1:2] = W*Diagonal([0.3, -0.5])/W + Ω[1:2, 1:2]
    A[3:4, 3:4] = [-0.2 0.9; -0.4 -0.2]

    L2(t, u, y, dydt) = mul!(dydt, rotation(ω, t)*A*rotation(ω, t)', y)
    out = floquet(L2, n, 1e-3, zero_orbit(n, T); λtol=1e-11, maxiter=100)
    M   = exp(Ω*T)*exp((A - Ω)*T)
    μs  = log.(complex.(eigvals(M)))./T
    @test out.converged
    @test distance(out.exponents, μs, T) < 1e-8
    @test distance(out.exponents, [0.3 + im*π/T, -0.5 + im*π/T, -0.2 + 0.6im, -0.2 - 0.6im], T) < 1e-8
end


@testset "shift for relative periodic orbits     " begin
    # x' = C x, with C commuting with the rotation S by an angle φ in the
    # (x1, x2) plane. Applying S at the end of each period gives the
    # multipliers of S exp(C T), with exponents a ± i(b + φ/T) and c.
    T = 3.0
    a = 0.2
    b = 0.5
    c = -0.7
    φ = 0.9
    C = [a -b 0; b a 0; 0 0 c]
    S = [cos(φ) -sin(φ) 0; sin(φ) cos(φ) 0; 0 0 1]

    L(t, u, y, dydt) = mul!(dydt, C, y)
    out = floquet(L, 3, 1e-3, zero_orbit(3, T); shift=q -> (q .= S*q; q),
                  λtol=1e-11, maxiter=100)
    @test out.converged
    @test distance(out.exponents, [a + im*(b + φ/T), a - im*(b + φ/T), c], T) < 1e-8
end


# ---------------------------------------------------------------------------- #
# LORENZ PERIODIC ORBITS

@testset "Lorenz orbits AB and AAB               " begin
    # guesses on the section z = 27, from Viswanath (2003) for AB, and from a
    # near recurrence of a long trajectory for AAB
    guesses = ((x0 = [-13.76, -19.58, 27.0], T = 1.5587),
               (x0 = [ 14.43,  21.11, 27.0], T = 2.3055))
    periods = (1.5586522, 2.3059073)

    for (guess, T_ref) in zip(guesses, periods)
        # converge the orbit
        n     = 4000
        x0, T = shoot(guess.x0, guess.T, n)
        @test abs(T - T_ref) < 1e-7

        # Floquet exponents along the stored orbit
        store = periodic_storage(lorenz, x0, T, n)
        out   = floquet(lorenz_tan, 3, T/n, store; λtol=1e-10, maxiter=50)
        λs    = out.exponents
        @test out.converged

        # one exponent is zero, for time translation
        @test abs(λs[2]) < 1e-6

        # the sum is the trace of the Jacobian
        @test abs(sum(real, λs) + 41/3) < 1e-6

        # the leading exponent agrees with the monodromy matrix of this short
        # orbit, including the sign of the multiplier
        _, M = monodromy(x0, T, n)
        μ    = real(eigvals(M)[argmax(abs.(eigvals(M)))])
        @test abs(real(λs[1]) - log(abs(μ))/T) < 1e-6
        @test cis(imag(λs[1])*T) ≈ sign(μ)

        # IMEX schemes along the same orbit, with a tolerance set by their order
        for (method, tol) in ((:CB4R3R4,  1e-8),
                              (:CB3R2R3e, 1e-6),
                              (:CB3R2R3c, 5e-6),
                              (:CB3R2R2,  1e-3),
                              (:CNRK2,    1e-3))
            out = floquet(lorenz_tan_ex, 3, T/n, store; A=Diagonal([-10.0, -1.0, -8/3]),
                          method=method, λtol=1e-10, maxiter=50)
            @test out.converged
            @test maximum(abs, out.exponents .- λs) < tol
        end
    end
end

@testset "Lorenz orbit with ten loops            " begin
    # guess on the section z = 27, from a near recurrence of a long trajectory
    n     = 20000
    x0, T = shoot([11.643, 14.899, 27.0], 7.4163, n)
    @test abs(T - 7.4162915) < 1e-7

    # Floquet exponents along the stored orbit
    store = periodic_storage(lorenz, x0, T, n)
    out   = floquet(lorenz_tan, 3, T/n, store; λtol=1e-10, maxiter=50)
    λs    = out.exponents
    @test out.converged

    # the zero exponent and the sum rule hold, although the stable multiplier,
    # about 1e-47, is far below the round-off of the monodromy matrix
    @test abs(λs[2]) < 1e-6
    @test abs(sum(real, λs) + 41/3) < 1e-6
end

@testset "RK8                                    " begin
    T     = T_AB
    store = periodic_storage(lorenz, x0_AB, T, 4000)
    if isdefined(Flows, :RK8)
        # with 500 steps per period the sum rule holds to round-off with
        # RK8, and only to the error of the scheme with RK4
        rk8 = floquet(lorenz_tan, 3, T/500, store; method=:RK8, λtol=1e-10)
        rk4 = floquet(lorenz_tan, 3, T/500, store; method=:RK4, λtol=1e-10)
        @test rk8.converged
        @test abs(sum(real, rk8.exponents) + 41/3) < 1e-12
        @test abs(sum(real, rk4.exponents) + 41/3) > 1e-8

        # the leading exponent agrees with RK4 with 4000 steps
        ref = floquet(lorenz_tan, 3, T/4000, store; λtol=1e-10)
        @test abs(rk8.exponents[1] - ref.exponents[1]) < 1e-7

        # the adjoint sweep with RK8
        out = floquet(lorenz_tan, 3, T/500, store; method=:RK8, λtol=1e-10,
                      vectors=:floquet, Ladj=lorenz_adj)
        @test out.converged
        @test out.exponents == rk8.exponents
        @test maximum(abs, out.adjointexponents .- out.exponents) < 1e-9
        @test maximum(k -> biorth(out.adjointvectors[k], out.floquetvectors[k]),
                      eachindex(out.times)) < 1e-12

        # RK8 is explicit
        @test_throws ArgumentError floquet(lorenz_tan, 3, T/500, store; method=:RK8,
                                           A=Diagonal([-10.0, -1.0, -8/3]))
    else
        @test_throws ArgumentError floquet(lorenz_tan, 3, T/500, store; method=:RK8)
        @info "this Flows.jl has no RK8: only the error for method = :RK8 was tested"
    end
end


# ---------------------------------------------------------------------------- #
# SCHUR VECTORS, FLOQUET VECTORS AND ADJOINTS

@testset "Schur vectors                          " begin
    n     = 4000
    T     = T_AB
    store = periodic_storage(lorenz, x0_AB, T, n)
    ref   = floquet(lorenz_tan, 3, T/n, store; λtol=1e-10)
    out   = floquet(lorenz_tan, 3, T/n, store; λtol=1e-10, vectors=:schur)

    # the exponents are those of the default call, bit for bit, and the
    # outputs that were not requested are nothing
    @test out.exponents == ref.exponents
    @test out.history == ref.history
    @test ref.bases === nothing
    @test out.floquetvectors === nothing

    # one basis at the start, one at each step and one at the end
    @test length(out.times) == n + 1
    @test length(out.bases) == n + 1
    @test size(out.stretching) == (3, n + 1)
    @test out.times[1] == 0
    @test abs(out.times[end] - T) < 1e-12

    # orthonormal bases
    gram(Q) = [dot(a, b) for a in Q, b in Q]
    @test maximum(k -> maximum(abs, gram(out.bases[k]) - I), (1, n ÷ 2, n + 1)) < 1e-13

    # the stretching vanishes at the start and adds up to the exponents
    @test maximum(abs, out.stretching[:, 1]) < 1e-13
    @test maximum(i -> abs(sum(out.stretching[out.order[i], :])/T - real(out.exponents[i])), 1:3) < 1e-10

    # every 7 steps: the bases at the common times are the same, because the
    # QR factorisation of the tangent map is unique
    out7 = floquet(lorenz_tan, 3, T/n, store; λtol=1e-10, vectors=:schur, oneevery=7)
    K7   = cld(n, 7)
    @test length(out7.times) == K7 + 1
    @test maximum(abs, out7.exponents .- out.exponents) < 1e-11
    @test maximum(k -> maximum(norm.(out7.bases[k] .- out.bases[7(k-1)+1])), 1:K7) < 1e-12
    @test maximum(norm.(out7.bases[end] .- out.bases[end])) < 1e-12
end

@testset "Floquet vectors, Lorenz orbit AB       " begin
    n     = 4000
    T     = T_AB
    dt    = T/n
    store = periodic_storage(lorenz, x0_AB, T, n)
    ref   = floquet(lorenz_tan, 3, dt, store; λtol=1e-10)
    out   = floquet(lorenz_tan, 3, dt, store; λtol=1e-10, vectors=:floquet, Ladj=lorenz_adj)
    ts    = out.times
    Φ     = out.floquetvectors
    Ψ     = out.adjointvectors

    # the forward exponents are unchanged, and the adjoint ones agree
    @test out.converged
    @test out.exponents == ref.exponents
    @test maximum(abs, out.adjointexponents .- out.exponents) < 1e-9

    # biorthogonal at every time, and of unit norm
    @test maximum(k -> biorth(Ψ[k], Φ[k]), eachindex(ts)) < 1e-12
    @test maximum(k -> maximum(abs, norm.(Φ[k]) .- 1), eachindex(ts)) < 1e-13

    # the Floquet vector of the zero exponent is the direction of the flow
    u   = zeros(3)
    fΦ2 = maximum(eachindex(ts)) do k
        offspan(lorenz(0.0, store(u, ts[k]), zeros(3)), Φ[k][out.order[2]])
    end
    @test fΦ2 < 1e-8

    # covariant over one segment, forward for φ and backward for ψ
    segments = Iterators.product((1, 1000, 2500, 3999), 1:3)
    @test maximum(segments) do (k, j)
        offspan(propagate(lorenz_tan, dt, store, Φ[k][j], (ts[k], ts[k+1])), Φ[k+1][j])
    end < 1e-12
    @test maximum(segments) do (k, j)
        offspan(propagate(lorenz_adj, dt, store, Ψ[k+1][j], (ts[k+1], ts[k]); backward=true), Ψ[k][j])
    end < 1e-12

    # over the period each vector is multiplied by its multiplier, which the
    # adjoint recovers. The residual is relative to the leading multiplier,
    # because the stable vector is conditioned by Λ₁/Λ₃.
    Λ = real.(exp.(out.exponents .* T))
    for i = 1:3
        j  = out.order[i]
        Mφ = propagate(lorenz_tan, dt, store, Φ[1][j], (0.0, T))
        @test abs(dot(Ψ[1][j], Mφ) - Λ[i]) < 1e-8*abs(Λ[i])
        @test norm(Mφ - Λ[i]*Φ[1][j]) < 1e-9*abs(Λ[1])
    end
end

@testset "Floquet vectors, IMEX scheme           " begin
    # the implicit diagonal is self-adjoint, the adjoint of the explicit
    # part goes in Ladj
    n     = 4000
    T     = T_AB
    store = periodic_storage(lorenz, x0_AB, T, n)
    out   = floquet(lorenz_tan_ex, 3, T/n, store; A=Diagonal([-10.0, -1.0, -8/3]),
                    method=:CB4R3R4, λtol=1e-10, vectors=:floquet, Ladj=lorenz_adj_ex)
    ts    = out.times
    Φ     = out.floquetvectors
    Ψ     = out.adjointvectors
    u     = zeros(3)
    @test out.converged
    @test maximum(abs, out.adjointexponents .- out.exponents) < 1e-8
    @test maximum(k -> biorth(Ψ[k], Φ[k]), eachindex(ts)) < 1e-12
    @test maximum(eachindex(ts)) do k
        offspan(lorenz(0.0, store(u, ts[k]), zeros(3)), Φ[k][out.order[2]])
    end < 1e-7
end

@testset "Floquet vectors, constant coefficients " begin
    # x' = A x with eigenvalues 0.5, 0.1 ± 2i, -0.2 and -1.3 in the basis V.
    # The Floquet vectors are the eigenvectors of A, the columns of V, with
    # the pair in the plane of the first two. The adjoints are those of Aᵀ,
    # the rows of V⁻¹.
    rng = MersenneTwister(1)
    n   = 5
    D   = zeros(n, n)
    D[1:2, 1:2] = [0.1 -2.0; 2.0 0.1]
    D[3, 3] = 0.5
    D[4, 4] = -0.2
    D[5, 5] = -1.3
    V   = randn(rng, n, n)
    A   = V*D/V
    W   = Matrix(transpose(inv(V)))
    T   = 3.7
    L(t, u, y, dydt)    = mul!(dydt, A, y)
    Ladj(t, u, w, dwdt) = mul!(dwdt, transpose(A), w)
    store = zero_orbit(n, T)

    # the columns of V and W for the exponent λ
    function columns(λ)
        abs(imag(λ)) > 1e-6 && return 1:2
        c = (3, 4, 5)[argmin(abs.(real(λ) .- (0.5, -0.2, -1.3)))]
        return c:c
    end

    out = floquet(L, n, 1e-3, store; λtol=1e-11, maxiter=100, vectors=:floquet, Ladj=Ladj)
    ts  = out.times
    Φ   = out.floquetvectors
    Ψ   = out.adjointvectors
    p   = out.pairs[1]
    @test out.converged
    @test length(out.pairs) == 1
    @test maximum(abs, out.adjointexponents .- out.exponents) < 1e-9
    @test maximum(Iterators.product(eachindex(ts), 1:n)) do (k, i)
        offspan(Φ[k][out.order[i]], V[:, columns(out.exponents[i])])
    end < 1e-9
    @test maximum(Iterators.product(eachindex(ts), 1:n)) do (k, i)
        offspan(Ψ[k][out.order[i]], W[:, columns(out.exponents[i])])
    end < 1e-9
    @test maximum(k -> abs(dot(Φ[k][p], Φ[k][p+1])), eachindex(ts)) < 1e-13
    @test maximum(k -> biorth(Ψ[k], Φ[k]), eachindex(ts)) < 1e-12

    # the basis of the plane of the pair is the one closest to the Schur
    # vectors of the pair at every time, and it varies continuously along the
    # orbit, like them, here by less than 1e-2 between times 1e-3 apart
    B(k) = [dot(Φ[k][i], out.bases[k][j]) for i in p:p+1, j in p:p+1]
    @test maximum(k -> norm(B(k) - B(k)'), eachindex(ts)) < 1e-12
    @test minimum(k -> minimum(eigvals(Symmetric(B(k)))), eachindex(ts)) > 0
    @test maximum(Iterators.product(1:length(ts)-1, p:p+1)) do (k, j)
        norm(Φ[k+1][j] - Φ[k][j])
    end < 1e-2

    # over the period: the real vectors are multiplied by their multipliers,
    # and the plane of the pair is invariant, with the multipliers of the pair
    MΦ = [propagate(L, 1e-3, store, φ, (0.0, T)) for φ in Φ[1]]
    for i in findall(λ -> imag(λ) == 0, out.exponents)
        j = out.order[i]
        Λ = exp(real(out.exponents[i])*T)
        @test norm(MΦ[j] - Λ*Φ[1][j]) < 1e-9*Λ
    end
    B  = [dot(Ψ[1][i], MΦ[j]) for i in p:p+1, j in p:p+1]
    Λp = exp((0.1 + 2im)*T)
    @test maximum(j -> offspan(MΦ[j], hcat(Φ[1][p], Φ[1][p+1])), p:p+1) < 1e-9
    @test maximum(abs, sort(eigvals(B); by=imag) .- [conj(Λp), Λp]) < 1e-9*abs(Λp)

    # the three leading vectors only, from random initial vectors
    init = [randn(rng, n) for i = 1:3]
    out  = floquet(L, 3, 1e-3, store; init=init, λtol=1e-11, maxiter=100,
                   vectors=:floquet, Ladj=Ladj)
    @test out.converged
    @test maximum(Iterators.product(eachindex(out.times), 1:3)) do (k, i)
        offspan(out.floquetvectors[k][out.order[i]], V[:, columns(out.exponents[i])])
    end < 1e-9
end

@testset "Floquet vectors, negative multipliers  " begin
    # the time-periodic system (b) above, with a fixed W. The multipliers of
    # the rotating plane are negative, and the vectors there turn with R(t).
    # The basis converges to the order 0.3 + iπ/T, -0.5 + iπ/T, pair, which
    # is not the order of the sorted exponents.
    n = 4
    T = 2.0
    ω = π/T
    rotation(t) = [cos(ω*t) -sin(ω*t) 0 0;
                   sin(ω*t)  cos(ω*t) 0 0;
                   0         0        1 0;
                   0         0        0 1]
    Ω = [0 -ω 0 0; ω 0 0 0; 0 0 0 0; 0 0 0 0]
    W = [1.0 0.3; -0.2 0.8]
    A = zeros(n, n)
    A[1:2, 1:2] = W*Diagonal([0.3, -0.5])/W + Ω[1:2, 1:2]
    A[3:4, 3:4] = [-0.2 0.9; -0.4 -0.2]
    L(t, u, y, dydt)    = mul!(dydt, rotation(t)*A*rotation(t)', y)
    Ladj(t, u, w, dwdt) = mul!(dwdt, rotation(t)*A'*rotation(t)', w)
    store = zero_orbit(n, T)

    out = floquet(L, n, 1e-3, store; λtol=1e-11, maxiter=100, vectors=:floquet, Ladj=Ladj)
    ts  = out.times
    Φ   = out.floquetvectors
    Ψ   = out.adjointvectors
    @test out.converged
    @test out.order == [1, 3, 4, 2]
    @test out.pairs == [3]
    @test maximum(eachindex(ts)) do k
        max(offspan(Φ[k][1], rotation(ts[k])*[W[:, 1]; 0; 0]),
            offspan(Φ[k][2], rotation(ts[k])*[W[:, 2]; 0; 0]))
    end < 1e-9
    @test maximum(k -> max(offspan(Φ[k][3], I[1:4, 3:4]), offspan(Φ[k][4], I[1:4, 3:4])), eachindex(ts)) < 1e-12
    @test maximum(k -> biorth(Ψ[k], Φ[k]), eachindex(ts)) < 1e-12

    # over the period the two real vectors change sign
    for i in (1, 4)
        j  = out.order[i]
        Λ  = real(exp(out.exponents[i]*T))
        Mφ = propagate(L, 1e-3, store, Φ[1][j], (0.0, T))
        @test Λ < 0
        @test norm(Mφ - Λ*Φ[1][j]) < 1e-9*abs(Λ)
    end
end


# ---------------------------------------------------------------------------- #
# RELATIVE PERIODIC ORBITS

@testset "Floquet vectors, shift                 " begin
    # the system of the testset of the shift: the pair lies in the plane
    # (x1, x2) and the real vector along x3. The adjoint shift is Sᵀ.
    T = 3.0
    C = [0.2 -0.5 0; 0.5 0.2 0; 0 0 -0.7]
    S = [cos(0.9) -sin(0.9) 0; sin(0.9) cos(0.9) 0; 0 0 1]
    L(t, u, y, dydt)    = mul!(dydt, C, y)
    Ladj(t, u, w, dwdt) = mul!(dwdt, transpose(C), w)
    out = floquet(L, 3, 1e-3, zero_orbit(3, T); shift=q -> (q .= S*q; q),
                  shiftadj=q -> (q .= S'*q; q), λtol=1e-11, maxiter=100,
                  vectors=:floquet, Ladj=Ladj)
    ts  = out.times
    Φ   = out.floquetvectors
    @test out.converged
    @test out.pairs == [1]
    @test maximum(eachindex(ts)) do k
        max(offspan(Φ[k][1], I[1:3, 1:2]), offspan(Φ[k][2], I[1:3, 1:2]), offspan(Φ[k][3], I[1:3, 3]))
    end < 1e-12
    @test maximum(k -> biorth(out.adjointvectors[k], Φ[k]), eachindex(ts)) < 1e-12

    # a shift S that does not commute with C. The Floquet vectors at t are
    # exp(C t) times the eigenvectors of the monodromy S exp(C T), and the
    # adjoints exp(-Cᵀ t) times the eigenvectors of its transpose. Pairs are
    # compared through the real plane of their eigenvectors.
    rng = MersenneTwister(3)
    D   = zeros(4, 4)
    D[1, 1]     = 0.3
    D[2:3, 2:3] = [-0.1 -0.7; 0.7 -0.1]
    D[4, 4]     = -0.6
    Z   = randn(rng, 4, 4)
    C   = Z*D/Z
    S   = Matrix(qr(randn(rng, 4, 4)).Q)
    T   = 2.5
    M   = S*exp(C*T)
    Er  = eigen(M)
    El  = eigen(Matrix(transpose(M)))
    L2(t, u, y, dydt)    = mul!(dydt, C, y)
    L2adj(t, u, w, dwdt) = mul!(dwdt, transpose(C), w)
    store = zero_orbit(4, T)
    out   = floquet(L2, 4, 1e-3, store; shift=q -> (q .= S*q; q),
                    shiftadj=q -> (q .= S'*q; q), λtol=1e-10, maxiter=100,
                    vectors=:floquet, Ladj=L2adj)
    ts    = out.times

    # the eigenvector, or the real plane of a pair, of the multiplier exp(λT)
    function eigenbasis(E, λ)
        j = argmin(abs.(E.values .- exp(λ*T)))
        v = E.vectors[:, j]
        return isreal(E.values[j]) ? real(v) : hcat(real(v), imag(v))
    end

    @test out.converged
    @test maximum(Iterators.product((1, (length(ts) + 1) ÷ 2, length(ts)), 1:4)) do (k, i)
        j = out.order[i]
        max(offspan(out.floquetvectors[k][j], exp(C*ts[k])*eigenbasis(Er, out.exponents[i])),
            offspan(out.adjointvectors[k][j], exp(-C'*ts[k])*eigenbasis(El, out.exponents[i])))
    end < 1e-9
    @test maximum(k -> biorth(out.adjointvectors[k], out.floquetvectors[k]), eachindex(ts)) < 1e-12

    # the multipliers follow from the vectors at the start, ⟨Ψ, M Φ⟩, by blocks
    Φ0     = out.floquetvectors[1]
    Ψ0     = out.adjointvectors[1]
    MΦ     = [S*propagate(L2, 1e-3, store, φ, (0.0, T)) for φ in Φ0]
    blocks = [i in out.pairs ? (i:i+1) : (i:i) for i = 1:4 if i-1 ∉ out.pairs]
    Λs     = reduce(vcat, [eigvals([dot(Ψ0[i], MΦ[j]) for i in b, j in b]) for b in blocks])
    @test maximum(Λ -> minimum(abs.(Λ .- Er.values)), Λs) < 1e-9*maximum(abs, Er.values)
end

@testset "relative periodic orbit                " begin
    # the orbit AB of the Lorenz equations with a = 1, which rotates a by ℓ
    # over the period. The storage holds both ends of the period.
    n     = 4000
    T     = T_AB
    u0    = [x0_AB; 1.0; 0.0]
    store = RAMStorage(zeros(5); degree=5)
    uT    = flow(lorenzphase, RK4(zeros(5)), TimeStepConstant(T/n))(copy(u0), (0, T), store)
    ℓ     = atan(uT[5], uT[4])
    R(θ)  = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    back!(v)  = (v[4:5] .= R(-ℓ)*v[4:5]; v)
    forth!(v) = (v[4:5] .= R(ℓ)*v[4:5]; v)

    out = floquet(lorenzphase_tan, 5, T/n, store; shift=back!, shiftadj=forth!, λtol=1e-10,
                  maxiter=100, vectors=:floquet, Ladj=lorenzphase_adj)
    ts  = out.times
    Φ   = out.floquetvectors

    # the exponents of the Lorenz orbit, 0 for the rotation and -2 for the radius
    @test out.converged
    @test maximum(abs, out.exponents .- [0.994650025031732, 0, 0, -2, -14.6613166914551]) < 1e-7
    @test abs(sum(real, out.exponents) + 41/3 + 2) < 1e-7
    @test maximum(abs, out.adjointexponents .- out.exponents) < 1e-8
    @test maximum(k -> biorth(out.adjointvectors[k], Φ[k]), eachindex(ts)) < 1e-12

    # the direction of the flow and that of the rotation, (0, 0, 0, -q, p),
    # lie in the plane of the two zero exponents, and the Floquet vector of
    # the exponent -2 is radial, along (0, 0, 0, p, q)
    u       = zeros(5)
    neutral = k -> hcat(Φ[k][out.order[2]], Φ[k][out.order[3]])
    @test maximum(eachindex(ts)) do k
        store(u, ts[k])
        max(offspan(lorenzphase(0.0, u, zeros(5)), neutral(k)),
            offspan([0, 0, 0, -u[5], u[4]], neutral(k)))
    end < 1e-8
    @test maximum(eachindex(ts)) do k
        store(u, ts[k])
        offspan(Φ[k][out.order[4]], [0, 0, 0, u[4], u[5]])
    end < 1e-8
end

@testset "verbose                                " begin
    # one line per period of each sweep, and the same results as without
    T     = T_AB
    store = periodic_storage(lorenz, x0_AB, T, 4000)
    quiet = floquet(lorenz_tan, 3, T/4000, store; λtol=1e-10, vectors=:floquet, Ladj=lorenz_adj)
    path, io = mktemp()
    loud  = redirect_stdout(io) do
        floquet(lorenz_tan, 3, T/4000, store; λtol=1e-10, vectors=:floquet, Ladj=lorenz_adj,
                verbose=true)
    end
    close(io)
    lines = readlines(path)
    @test count(startswith("period"), lines) == loud.periods
    @test count(startswith("adjoint"), lines) ≥ 2
    @test loud.exponents == quiet.exponents
    @test loud.adjointexponents == quiet.adjointexponents
    @test loud.floquetvectors == quiet.floquetvectors
end
