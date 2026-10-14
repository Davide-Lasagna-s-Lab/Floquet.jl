using LinearAlgebra
using Random
using Flows
using Lyapunov
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


# ---------------------------------------------------------------------------- #
# BUILDING BLOCKS

@testset "procrustes                             " begin
    N = 5
    B = rand(2*N, N)
    Q_exact = Matrix(qr(rand(N, N)).Q)
    A = B*Q_exact

    as = couple([A[:, i] for i = 1:N]...)
    bs = couple([B[:, i] for i = 1:N]...)

    Q = Lyapunov.procrustes(as, bs, false)
    @test norm(Q_exact .- Q) < 1e-14
    @test norm(A - B*Q) < 1e-14
end

@testset "pair_locations                         " begin
    N = 5
    B = zeros(N, N) + I
    @test Lyapunov.pair_locations(B, 1e-6) == Int[]
    @test Lyapunov.pair_locations(ones(1, 1), 1e-6) == Int[]

    θ = π/4; c, s = cos(θ), sin(θ)
    B = zeros(N, N) + I
    B[2, 2] = c; B[3, 3] = c; B[2, 3] = s; B[3, 2] = -s
    @test Lyapunov.pair_locations(B, 1e-6) == Int[2]

    B = zeros(N, N) + I
    B[2, 2] = -c; B[3, 3] = c; B[2, 3] = s; B[3, 2] = -s
    @test Lyapunov.pair_locations(B, 1e-6) == Int[]

    B = [0.9999999999994441  1.0546919551626985e-6;
        -1.05469195515357e-6 0.9999999999989968]
    @test Lyapunov.pair_locations(B, 1e-6) == Int[1]
    @test Lyapunov.pair_locations(B, 1e-5) == Int[]

    B = [0.9999999999999919    1.2968516247787576e-7;
        -1.2968516246475442e-7 0.999999999999988]
    @test Lyapunov.pair_locations(B, 1e-7) == Int[1]
    @test Lyapunov.pair_locations(B, 1e-6) == Int[]
    @test Lyapunov.pair_locations(B, 1e-5) == Int[]

    B = [1    -1.1e-4;
         1.1e-4  1]
    @test Lyapunov.pair_locations(B,   1e-4) == Int[1]
    @test Lyapunov.pair_locations(B, 1.1e-4) == Int[]
end

@testset "gs!                                    " begin
    for N = (3, 5, 10)
        A1 = randn(N, N)
        A2 = copy(A1)
        f  = qr(A1)
        r1 = Lyapunov.gs!(couple(nothing, [view(A1, :, i) for i = 1:N]...), true)
        r2 = Lyapunov.gs!(couple([view(A2, :, i) for i = 1:N]...), false)

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
