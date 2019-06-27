using LinearAlgebra
using Statistics
using Lyapunov
using Random
using Flows
using Test

@testset "gsortho!                               " begin
    Random.seed!(0)
    for N = (3, 5, 10)
        A  = randn(N, N)
        f = qr(A, Val(false))
        r = gsortho!(couple(nothing, [view(A, :, i) for i = 1:N]...))

        # check diagonal
        for i = 1:N
            @test abs(abs(r[i]) - abs(f.R[i, i])) < 1e-13
        end

        # check orthogonality of vectors
        @test maximum(abs, A'*A - I) < 1e-13
    end
end

@testset "utils                                  " begin
    # call dependencies for 3 linearised equations plus nonlinear problem
    deps = calldeps(4)
    @test deps[1] == (1, )
    @test deps[2] == (1, 2)
    @test deps[3] == (1, 3)
    @test deps[4] == (1, 4)
    @test_throws BoundsError deps[5]

    # initial conditions for three tangent vectors
    xs = init_IC(4, [2.0, 2.0, 2.0, 2.0, 2.0])
    @test xs[1] == [2, 2, 2, 2, 2]
    @test xs[2] == [1, 0, 0, 0, 0]
    @test xs[3] == [0, 1, 0, 0, 0]
    @test xs[4] == [0, 0, 1, 0, 0]
    @test length(xs) == 4

    # test opcouple
    ops = opcouple(4, +, -)
    @test ops[1] == +
    @test ops[2] == -
    @test ops[3] == -
    @test ops[4] == -
    length(ops) == 4
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
    N = 4

    # time step
    dt = 1e-2

    # call dependency
    deps = calldeps(N)

    # initial conditions
    xs = couple(init_IC(N, x)...)

    # coupled operator
    ϕ = flow(opcouple(N, Lorenz, LorenzTan),
             deps,
             RK4(couplecopy(N, x), :NORMAL),
             TimeStepConstant(dt))

    # try propagating a bit (this should not allocate)
    fun(ϕ, xs) = @allocated ϕ(xs, (0, 1))
    @test fun(ϕ, xs) == 0
    
    # define monitor
    mon = Monitor(xs, gsortho!; oneevery=10)

    # run
    ϕ(xs, (0, 10000), mon)
    ϕ(xs, (0, 10000), mon)
    ϕ(xs, (0, 10000), mon)
    ϕ(xs, (0, 10000), mon)

    # calculate mean exponents
    λ0, λ1, λ2 = mean.(process(samples(mon), 10*dt))

    @test   0.900 < λ0 <   0.910
    @test  -0.001 < λ1 <   0.001
    @test -14.600 < λ2 < -14.500
end

