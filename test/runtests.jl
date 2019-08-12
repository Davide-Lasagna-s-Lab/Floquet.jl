using LinearAlgebra
using Statistics
using Lyapunov
using Random
using Flows
using Test

Random.seed!(0)

@testset "procrustes                             " begin
    N = 5
    B = rand(2*N, N)
    Q_exact = qr(rand(N, N)).Q; 
    A = B*Q_exact
    
    as = couple([A[:, i] for i = 1:N]...)
    bs = couple([B[:, i] for i = 1:N]...)

    Q = procrustes(as, bs, false)
    @test norm(Q_exact .- Q) < 2e-15
    @test norm(A - B*Q) < 2e-15
end

# @testset "tridiagdot!                            " begin
#     N = 5
#     B = Tridiagonal(zeros(N-1), zeros(N), zeros(N-1))
#     Qx = qr(rand(10, 10)).Q; xs = couple([Qx[:, i] for i = 1:N]...)
#     Qy = qr(rand(10, 10)).Q; ys = couple([Qy[:, i] for i = 1:N]...)
#     tridiagdot!(B, xs, ys, false)

#     @test B[1, 1] == dot(xs[1], ys[1])
#     @test B[2, 3] == dot(xs[2], ys[3])
#     @test B[3, 2] == dot(xs[3], ys[2])
# end

@testset "pair_locations                         " begin
    N = 5
    B = zeros(N, N) + I
    @test pair_locations(B, 1e-6) == Int[]

    θ = π/4; c, s = cos(θ), sin(θ)
    B = zeros(N, N) + I
    B[2, 2] = c; B[3, 3] = c; B[2, 3] = s; B[3, 2] = -s
    @test pair_locations(B, 1e-6) == Int[2]
    
    B = zeros(N, N) + I
    B[2, 2] = -c; B[3, 3] = c; B[2, 3] = s; B[3, 2] = -s
    @test pair_locations(B, 1e-6) == Int[]
    
    B = [0.9999999999994441  1.0546919551626985e-6;
        -1.05469195515357e-6 0.9999999999989968]
    @test pair_locations(B, 1e-6) == Int[1]
    @test pair_locations(B, 1e-5) == Int[]

    B = [0.9999999999999919    1.2968516247787576e-7;
        -1.2968516246475442e-7 0.999999999999988]
    @test pair_locations(B, 1e-7) == Int[1]
    @test pair_locations(B, 1e-6) == Int[]
    @test pair_locations(B, 1e-5) == Int[]

    B = [1    -1.1e-4;
         1.1e-4  1]
    @test pair_locations(B,   1e-4) == Int[1]
    @test pair_locations(B, 1.1e-4) == Int[]
end

@testset "rfactor!                               " begin
    Random.seed!(0)
    for N = (3, 5, 10)
        A1  = randn(N, N)
        A2 = copy(A1)
        f = qr(A1, Val(false))
        r1 = rfactor!(couple(nothing, [view(A1, :, i) for i = 1:N]...), true)
        r2 = rfactor!(couple([view(A2, :, i) for i = 1:N]...), false)

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

@testset "utils                                  " begin
    # call dependencies for 3 linearised equations plus nonlinear problem
    deps = calldeps(3)
    @test deps[1] == (1, )
    @test deps[2] == (1, 2)
    @test deps[3] == (1, 3)
    @test deps[4] == (1, 4)
    @test_throws BoundsError deps[5]

    # call dependencies for 3 linearised equations
    deps = calldeps(3, false)
    @test deps[1] == (1, )
    @test deps[2] == (2, )
    @test deps[3] == (3, )
    @test_throws BoundsError deps[4]

    # initial conditions for four tangent vectors
    xs = init_IC(4, [2.0, 2.0, 2.0, 2.0, 2.0])
    @test xs[1] == [1, 0, 0, 0, 0]
    @test xs[2] == [0, 1, 0, 0, 0]
    @test xs[3] == [0, 0, 1, 0, 0]
    @test xs[4] == [0, 0, 0, 1, 0]
    @test length(xs) == 4

    # test opcouple
    ops = opcouple(4, +, -)
    @test ops[1] == +
    @test ops[2] == -
    @test ops[3] == -
    @test ops[4] == -
    @test ops[5] == -
    @test length(ops) == 5

    ops = opcouple(4, *)
    @test ops[1] == *
    @test ops[2] == *
    @test ops[3] == *
    @test ops[4] == *
    @test length(ops) == 4
end

function Lorenz(t::Real, u::V, dudt::V) where {V <: AbstractVector}
    x, y, z = u
    dudt[1] =   10 * (y - x)
    dudt[2] =   28 *  x - y - x*z
    dudt[3] = -8/3 * z + x*y
    return dudt
end

function LorenzTan(t::Real, u::V, dudt::V, v::V, dvdt::V) where {V <: AbstractVector}
    x′, y′, z′ = v
    x,  y,  z  = u
    dvdt[1] =  10 * (y′ - x′)
    dvdt[2] =  (28-z)*x′ - y′ - x*z′
    dvdt[3] = -8/3*z′ + x*y′ + x′*y
    return dvdt
end

@testset "Lorenz                                 " begin

    # initial condition
    x = Float64[9.1419853, 1.648665, 35.21793]

    # one nonlinear equation and 3 linearised systems
    N = 3

    # time step
    dt = 1e-3

    # call dependency
    deps = calldeps(N)

    # initial conditions
    xs = couple(x, init_IC(N, x)...)

    # coupled operator
    ϕ = flow(opcouple(N, Lorenz, LorenzTan),
             deps,
             RK4(couplecopy(N+1, x)),
             TimeStepConstant(dt))

    # try propagating a bit (this should not allocate)
    fun(ϕ, xs) = @allocated ϕ(xs, (0, 1))
    @test fun(ϕ, xs) == 0
    
    # define monitor
    mon = Monitor(xs, x->tuple(diag(rfactor!(x, true))...); oneevery=10)

    # run
    ϕ(xs, (0, 100), mon)
    ϕ(xs, (0, 100), mon)
    ϕ(xs, (0, 100), mon)
    ϕ(xs, (0, 100), mon)

    # calculate mean exponents
    λ0, λ1, λ2 = mean.(process(samples(mon), times(mon)))
    # println(λ0)
    # println(λ1)
    # println(λ2)

    # @test   0.900 < λ0 <   0.910
    # @test  -0.001 < λ1 <   0.001
    # @test -14.600 < λ2 < -14.500
end

