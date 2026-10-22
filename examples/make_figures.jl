# Makes the figures of the README in assets/ and prints every number that the README quotes: the
# table of the sum rule on three periodic orbits of the Lorenz equations, the worked example on the
# orbit AB and its convergence, the stretching and the Floquet vectors along AB, and a relative
# periodic orbit of the Lorenz equations coupled to a complex amplitude.
#
#     julia --project=examples -e 'using Pkg; Pkg.instantiate()'
#     julia --project=examples examples/make_figures.jl
#
# The scheme :RK8 needs Flows.jl with RK8, on the branch rk8 until it is merged into master. The
# examples on the Kuramoto-Sivashinsky equation and on Kolmogorov flow are in the scripts
# kuramoto_sivashinsky.jl and kolmogorov.jl, and the colours and theme of the figures in style.jl.

using LinearAlgebra
using Printf
using Flows
using Floquet
using CairoMakie

const ASSETS = joinpath(@__DIR__, "..", "assets")


# ============================================================================ #
# The Lorenz equations                                                         #
# ============================================================================ #

# Lorenz equations, with σ = 10, ρ = 28 and β = 8/3
function lorenz(t, x, dxdt)
    dxdt[1] = 10*(x[2] - x[1])
    dxdt[2] = 28*x[1] - x[2] - x[1]*x[3]
    dxdt[3] = -8/3*x[3] + x[1]*x[2]
    return dxdt
end

# linearised equations, the Jacobian at u applied to the tangent vector y
function lorenz_tan(t, u, y, dydt)
    dydt[1] = 10*(y[2] - y[1])
    dydt[2] = (28 - u[3])*y[1] - y[2] - u[1]*y[3]
    dydt[3] = u[2]*y[1] + u[1]*y[2] - 8/3*y[3]
    return dydt
end

# the same, with the state first, as needed by a coupled integration
lorenz_tan(t, x, dxdt, y, dydt) = lorenz_tan(t, x, y, dydt)

# adjoint equations, the transpose of the Jacobian at u applied to w
function lorenz_adj(t, u, w, dwdt)
    dwdt[1] = -10*w[1] + (28 - u[3])*w[2] + u[2]*w[3]
    dwdt[2] = 10*w[1] - w[2] + u[1]*w[3]
    dwdt[3] = -u[1]*w[2] - 8/3*w[3]
    return dwdt
end

# explicit parts of both, for an IMEX scheme with A = Diagonal([-10.0, -1.0, -8/3])
function lorenz_tan_ex(t, u, y, dydt)
    dydt[1] = 10*y[2]
    dydt[2] = (28 - u[3])*y[1] - u[1]*y[3]
    dydt[3] = u[2]*y[1] + u[1]*y[2]
    return dydt
end

function lorenz_adj_ex(t, u, w, dwdt)
    dwdt[1] = (28 - u[3])*w[2] + u[2]*w[3]
    dwdt[2] = 10*w[1] + u[1]*w[3]
    dwdt[3] = -u[1]*w[2]
    return dwdt
end


# ============================================================================ #
# Periodic orbits by Newton shooting with RK8                                  #
# ============================================================================ #

# final state and monodromy matrix of the orbit through x0 over a time T, with n steps of RK8
function monodromy(x0, T, n)
    G = flow(couple(lorenz, lorenz_tan, lorenz_tan, lorenz_tan),
             CallDependency((1,), (1, 2), (1, 3), (1, 4)),
             RK8(couplecopy(4, zeros(3))),
             TimeStepConstant(T/n))
    z = couple(copy(x0), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    G(z, (0, T))
    return z[1], hcat(z[2], z[3], z[4])
end

# Newton shooting on the plane z = 27, the unknowns being x0[1], x0[2] and T. It stops when
# the residual no longer decreases, at the level of round-off.
function shoot(x0, T, n)
    x     = copy(x0)
    r_old = Inf
    for iter = 1:30
        xT, M = monodromy(x, T, n)
        r     = norm(xT .- x)
        r > r_old/2 && r < 1e-9 && return x, T, r
        r_old = r
        J     = hcat((M - I)[:, 1:2], lorenz(0.0, xT, zeros(3)))
        δ     = J \ (xT .- x)
        x[1] -= δ[1]
        x[2] -= δ[2]
        T    -= δ[3]
    end
    error("Newton shooting did not converge")
end

# one period of the orbit through x0, from n steps of RK8, in a periodic storage
function periodic_storage(x0, T, n)
    store = RAMStorage(zeros(3); period=T, storelast=false, degree=5)
    F     = flow(lorenz, RK8(zeros(3)), TimeStepConstant(T/n))
    F(copy(x0), (0, T), store)
    length(times(store)) == n || error("the storage holds a sample at t = T")
    return store
end

# the orbits of the table, with guesses on the plane z = 27 and the number of steps
orbits = [(name = "AB",       x0 = [-13.76, -19.58, 27.0],   T = 1.5587, n = 4000),
          (name = "AAB",      x0 = [ 14.43,  21.11, 27.0],   T = 2.3055, n = 4000),
          (name = "10 loops", x0 = [ 11.643, 14.899, 27.0],  T = 7.4163, n = 20000)]

cases = map(orbits) do o
    x0, T, r = shoot(o.x0, o.T, o.n)
    @printf "%-8s  x0 = [%.14f, %.14f, 27.0], T = %.15f, closure %.1e\n" o.name x0[1] x0[2] T r
    (name = o.name, x0 = x0, T = T, n = o.n, store = periodic_storage(x0, T, o.n))
end


# ============================================================================ #
# Table 1: the sum rule, from floquet and from the eigenvalues of M            #
# ============================================================================ #

println("\nTable 1")
println("| Orbit | Period T | Λ₃ | λ₃, floquet | λ₃, eigenvalues of M | error, floquet with RK4 | error, floquet with RK8 | error, eigenvalues of M |")
runs = map(cases) do o
    rk8 = floquet(lorenz_tan, 3, o.T/o.n, o.store; method=:RK8, λtol=1e-12, maxiter=100)
    rk4 = floquet(lorenz_tan, 3, o.T/o.n, o.store; method=:RK4, λtol=1e-12, maxiter=100)
    _, M = monodromy(o.x0, o.T, o.n)
    μs   = sort(log.(abs.(eigvals(M)))./o.T; rev=true)
    Λ3   = exp(real(rk8.exponents[3])*o.T)
    e4   = abs(sum(real, rk4.exponents) + 41/3)
    e8   = abs(sum(real, rk8.exponents) + 41/3)
    eM   = abs(sum(μs) + 41/3)
    @printf "| %s | %.3f | %.1e | %.6f | %.6f | %.1e | %.1e | %.1e |\n" o.name o.T Λ3 real(rk8.exponents[3]) μs[3] e4 e8 eM
    @printf "    converged %s / %s in %d / %d periods, λ₁ floquet %.12f, eig %.12f, λ₂ floquet %.1e, eig %.1e, |Λ₁/Λ₃| %.1e\n" rk8.converged rk4.converged rk8.periods rk4.periods real(rk8.exponents[1]) μs[1] real(rk8.exponents[2]) μs[2] exp((real(rk8.exponents[1]) - real(rk8.exponents[3]))*o.T)
    rk8
end


# ============================================================================ #
# The worked example of the README, on the orbit AB with RK4                   #
# ============================================================================ #

println("\nWorked example")

# a point of the orbit AB on the plane z = 27, and its period
x0 = [-13.76361068212564, -19.57875194244233, 27.0]
T  = 1.558652210720224

# one period of the orbit, in n steps of size T/n
n     = 4000
store = RAMStorage(zeros(3); period=T, storelast=false, degree=5)
F     = flow(lorenz, RK4(zeros(3)), TimeStepConstant(T/n))
F(copy(x0), (0, T), store)

# Floquet exponents, with three tangent vectors
out = floquet(lorenz_tan, 3, T/n, store; λtol=1e-10)

println("julia> out.exponents")
show(stdout, MIME"text/plain"(), out.exponents)
println("\n\njulia> out.converged, out.periods")
show(stdout, MIME"text/plain"(), (out.converged, out.periods))
println("\n\njulia> sum(real, out.exponents) + 41/3")
show(stdout, MIME"text/plain"(), sum(real, out.exponents) + 41/3)
println()

# the leading exponent against the monodromy matrix of this short orbit, with RK4 as in the tests
let G = flow(couple(lorenz, lorenz_tan, lorenz_tan, lorenz_tan),
             CallDependency((1,), (1, 2), (1, 3), (1, 4)),
             RK4(couplecopy(4, zeros(3))),
             TimeStepConstant(T/n))
    z = couple(copy(x0), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    G(z, (0, T))
    @printf "closure of the RK4 orbit %.1e, λ₁ against eig(M) with RK4 %.1e\n" norm(z[1] - x0) abs(real(out.exponents[1]) - log(maximum(abs, eigvals(hcat(z[2], z[3], z[4]))))/T)
end

# the same with the default tolerance, and the first lines printed with verbose = true
dflt = floquet(lorenz_tan, 3, T/n, store)
@printf "default λtol: %d periods, λ₂ = %.1e\n" dflt.periods real(dflt.exponents[2])
path, io = mktemp()
redirect_stdout(io) do
    floquet(lorenz_tan, 3, T/n, store; λtol=1e-10, verbose=true)
end
close(io)
foreach(println, readlines(path)[1:4])

# re-orthonormalisations every 1000 steps instead of every step
coarse = floquet(lorenz_tan, 3, T/n, store; λtol=1e-10, oneevery=1000)
@printf "oneevery = 1000: %d periods, |exponents - oneevery = 1| %.1e\n" coarse.periods maximum(abs, coarse.exponents .- out.exponents)

# the same orbit with an IMEX scheme, the diagonal of the Jacobian being implicit, with the vectors
imex = floquet(lorenz_tan_ex, 3, T/n, store; A=Diagonal([-10.0, -1.0, -8/3]), method=:CB4R3R4,
               λtol=1e-10, vectors=:floquet, Ladj=lorenz_adj_ex)
@printf "CB4R3R4: %d periods, |exponents - RK4| %.1e, |adjoint - forward| %.1e\n" imex.periods maximum(abs, imex.exponents .- out.exponents) maximum(abs, imex.adjointexponents .- imex.exponents)

# the orbit AB converged and stored with RK8, and its exponents with RK8, as in the table
ab  = cases[1]
ab8 = runs[1]
@printf "RK8: %d periods, exponents %.13f %.1e %.10f, sum rule %.1e\n" ab8.periods real(ab8.exponents[1]) real(ab8.exponents[2]) real(ab8.exponents[3]) sum(real, ab8.exponents) + 41/3


# ============================================================================ #
# Colours and theme of the figures                                             #
# ============================================================================ #

include(joinpath(@__DIR__, "style.jl"))


# ============================================================================ #
# Figure 1: convergence over the periods                                       #
# ============================================================================ #

println("\nFigure 1")
fig = Figure(size=(820, 340))
ax1 = Axis(fig[1, 1]; xlabel="period", ylabel=L"|\mathrm{Re}\,\lambda_2|", yscale=log10,
           yticks=LogTicks(-16:4:0))
ax2 = Axis(fig[1, 2]; xlabel="period", ylabel=L"|\lambda_1 + \lambda_2 + \lambda_3 + 41/3|",
           yscale=log10, yticks=LogTicks(-16:4:0))
for (o, r, c) in zip(cases, runs, SLOTS)
    p   = 1:r.periods
    λ2  = [max(abs(real(h[2])), 1e-16) for h in r.history]
    sr  = [max(abs(sum(real, h) + 41/3), 1e-16) for h in r.history]
    λ1T = real(r.exponents[1])*o.T

    # the rate |Λ₂/Λ₁| = exp(-λ₁ T) per period, from the second period down to round-off
    pr = [q for q in p if q ≥ 2 && λ2[2]*exp(-λ1T*(q - 2)) > 1e-15]
    lines!(ax1, pr, λ2[2] .* exp.(-λ1T .* (pr .- 2)); color=(c, 0.6), linestyle=:dash, linewidth=1.5)
    scatterlines!(ax1, p, λ2; color=c, label=o.name, markersize=8, strokecolor=:white, strokewidth=1.5)
    scatterlines!(ax2, p, sr; color=c, label=o.name, markersize=8, strokecolor=:white, strokewidth=1.5)
    text!(ax1, p[end] + 0.6, λ2[end]; text=o.name, align=(:left, :center), color=INK2, fontsize=13)
    @printf "%-8s %d periods, rate exp(-λ₁T) = %.3f, last |Re λ₂| %.1e, sum rule from %.1e to %.1e\n" o.name r.periods exp(-λ1T) λ2[end] minimum(sr) maximum(sr)
end
ylims!(ax1, 3e-17, 10)
ylims!(ax2, 3e-17, 10)
xlims!(ax1, 0, 25)
xlims!(ax2, 0, 25)
axislegend(ax2; position=:rt)
panel!(ax1, "(a)")
panel!(ax2, "(b)")
save(joinpath(ASSETS, "convergence.png"), fig; px_per_unit=2)


# ============================================================================ #
# Figure 2: stretching along the orbit AB                                      #
# ============================================================================ #

println("\nFigure 2")
sch = floquet(lorenz_tan, 3, ab.T/ab.n, ab.store; method=:RK8, λtol=1e-12, vectors=:schur)
fig = Figure(size=(820, 340))
ax  = Axis(fig[1, 1]; xlabel=L"t/T", ylabel="cumulative stretching", xticks=0:0.25:1)
cum = cumsum(sch.stretching; dims=2)
for (j, c, s) in zip(1:3, (UNSTABLE, NEUTRAL, STABLE), ("unstable", "neutral", "stable"))
    lines!(ax, sch.times ./ ab.T, cum[j, :]; color=c, label=s)
    scatter!(ax, [1.0], [cum[j, end]]; color=c, strokecolor=:white, strokewidth=2)
    v = @sprintf("%.3f", abs(cum[j, end]) < 5e-4 ? 0.0 : cum[j, end])
    text!(ax, 1.015, cum[j, end]; text=L"\lambda_%$j T = %$v", align=(:left, :center),
          color=INK2, fontsize=13)
    i = findfirst(==(j), sch.order)
    @printf "vector %d (%s): end %.12f, T Re λ %.12f, stretching over a step from %.1e to %.1e\n" j s cum[j, end] real(sch.exponents[i])*ab.T minimum(sch.stretching[j, 2:end]) maximum(sch.stretching[j, 2:end])
end
xlims!(ax, -0.02, 1.17)
axislegend(ax; position=:lb)
save(joinpath(ASSETS, "stretching.png"), fig; px_per_unit=2)


# ============================================================================ #
# Figure 3: Floquet vectors along the orbit AB                                 #
# ============================================================================ #

println("\nFigure 3")
fv = floquet(lorenz_tan, 3, ab.T/ab.n, ab.store; method=:RK8, λtol=1e-12, oneevery=10,
             vectors=:floquet, Ladj=lorenz_adj)
K  = length(fv.times) - 1
Φ  = fv.floquetvectors
Ψ  = fv.adjointvectors
u  = [ab.store(zeros(3), t) for t in fv.times]
f  = [lorenz(0.0, v, zeros(3)) for v in u]
φ1 = [Φ[k][fv.order[1]] for k = 1:K+1]
φ2 = [Φ[k][fv.order[2]] for k = 1:K+1]
φ3 = [Φ[k][fv.order[3]] for k = 1:K+1]

# angles between the vectors, which keep their orientation along the orbit
orientedangle(a, b) = acosd(clamp(dot(a, b)/(norm(a)*norm(b)), -1, 1))
θ12 = orientedangle.(φ1, φ2)
θ32 = orientedangle.(φ3, φ2)
θ13 = orientedangle.(φ1, φ3)
θf  = min.(orientedangle.(φ2, f), 180 .- orientedangle.(φ2, f))
θus = min.(θ13, 180 .- θ13)
bio = maximum(maximum(abs, [dot(ψ, φ) for ψ in Ψ[k], φ in Φ[k]] - I) for k = 1:K+1)
@printf "%d periods forward, converged %s, %d times\n" fv.periods fv.converged K + 1
@printf "smallest angle between the lines of φ₁ and φ₃ %.2f degrees at t/T = %.3f\n" minimum(θus) fv.times[argmin(θus)]/ab.T
@printf "oriented angles: φ₁ φ₂ from %.1f to %.1f, φ₃ φ₂ from %.1f to %.1f, φ₁ φ₃ from %.1f to %.1f\n" extrema(θ12)... extrema(θ32)... extrema(θ13)...
@printf "angle between φ₂ and f at most %.1e degrees\n" maximum(θf)
@printf "max |⟨ψᵢ, φⱼ⟩ - δᵢⱼ| %.1e, max |adjoint - forward exponents| %.1e\n" bio maximum(abs, fv.adjointexponents .- fv.exponents)
@printf "φ at the end against φ at the start: %.1e\n" maximum(j -> norm(Φ[end][j] - Φ[1][j]), 1:3)
@printf "exponents %s\nadjoint   %s\n" join([@sprintf("%+.13f", real(λ)) for λ in fv.exponents], " ") join([@sprintf("%+.13f", real(λ)) for λ in fv.adjointexponents], " ")

fig = Figure(size=(820, 420))
ax1 = Axis(fig[1, 1]; xlabel=L"x", ylabel=L"z", aspect=DataAspect())
ax2 = Axis(fig[1, 2]; xlabel=L"t/T", ylabel="angle (degrees)", yticks=0:45:180, xticks=0:0.25:1)
lines!(ax1, [v[1] for v in u], [v[3] for v in u]; color=AXIS, linewidth=2)
picks = round.(Int, range(1, K + 1; length=25))[1:24]
for k in picks, (φ, c) in ((φ3, STABLE), (φ1, UNSTABLE))
    # a segment through the point along the projection of the vector, on a ring of white
    a  = 2.2 .* (φ[k][1], φ[k][3])
    xs = [u[k][1] - a[1], u[k][1] + a[1]]
    zs = [u[k][3] - a[2], u[k][3] + a[2]]
    lines!(ax1, xs, zs; color=:white, linewidth=6)
    lines!(ax1, xs, zs; color=c, linewidth=2.5)
end
scatter!(ax1, [u[k][1] for k in picks], [u[k][3] for k in picks]; color=INK, markersize=5)
Legend(fig[0, 1], [LineElement(color=UNSTABLE, linewidth=2.5), LineElement(color=STABLE, linewidth=2.5)],
       [L"\varphi_1\ \mathrm{(unstable)}", L"\varphi_3\ \mathrm{(stable)}"]; orientation=:horizontal,
       framevisible=false, labelcolor=INK2, tellwidth=false)
lines!(ax2, fv.times ./ ab.T, θ12; color=UNSTABLE, label=L"\angle(\varphi_1, \varphi_2)")
lines!(ax2, fv.times ./ ab.T, θ32; color=STABLE, label=L"\angle(\varphi_3, \varphi_2)")
lines!(ax2, fv.times ./ ab.T, θ13; color=INK2, label=L"\angle(\varphi_1, \varphi_3)")
ylims!(ax2, 0, 180)
Legend(fig[0, 2], ax2; orientation=:horizontal, framevisible=false, labelcolor=INK2, tellwidth=false)
panel!(ax1, "(a)")
panel!(ax2, "(b)")
save(joinpath(ASSETS, "vectors.png"), fig; px_per_unit=2)


# ============================================================================ #
# A relative periodic orbit                                                    #
# ============================================================================ #

println("\nRelative periodic orbit")

# the Lorenz equations coupled to a complex amplitude a = p + iq, invariant under a → exp(iθ) a
const μ = 1.0
const ν = 0.05

function lorenzphase(t, u, dudt)
    x, y, z, p, q = u
    dudt[1] = 10*(y - x)
    dudt[2] = 28*x - y - x*z
    dudt[3] = -8/3*z + x*y
    dudt[4] = μ*p - ν*z*q - (p^2 + q^2)*p
    dudt[5] = μ*q + ν*z*p - (p^2 + q^2)*q
    return dudt
end

# its Jacobian, and the linearised and adjoint equations
function jacobian(u)
    x, y, z, p, q = u
    return [-10.0  10.0  0.0   0.0             0.0;
            28-z   -1.0  -x    0.0             0.0;
            y      x     -8/3  0.0             0.0;
            0.0    0.0   -ν*q  μ - 3p^2 - q^2  -ν*z - 2p*q;
            0.0    0.0   ν*p   ν*z - 2p*q      μ - p^2 - 3q^2]
end

lorenzphase_tan(t, u, y, dydt) = mul!(dydt, jacobian(u), y)
lorenzphase_adj(t, u, w, dwdt) = mul!(dwdt, transpose(jacobian(u)), w)

# the orbit AB converged with RK8, as for Table 1
x0, T, n = ab.x0, ab.T, ab.n

# one period of the orbit AB with a = √μ, in a storage with both ends, where x0, T and n
# are those of the orbit AB converged with RK8, as for Table 1
u0    = [x0; sqrt(μ); 0.0]
store = RAMStorage(zeros(5); degree=5)
F     = flow(lorenzphase, RK8(zeros(5)), TimeStepConstant(T/n))
uT    = F(copy(u0), (0, T), store)

# the angle by which a turns over the period, the shift back and its adjoint
ℓ = atan(uT[5], uT[4])
rotate!(v, θ) = (v[4:5] .= [cos(θ) -sin(θ); sin(θ) cos(θ)]*v[4:5]; v)

rpo = floquet(lorenzphase_tan, 5, T/n, store; method=:RK8, λtol=1e-12,
              shift    = v -> rotate!(v, -ℓ),
              shiftadj = v -> rotate!(v, ℓ),
              vectors  = :floquet,
              Ladj     = lorenzphase_adj)

closure = norm(rotate!(copy(uT), -ℓ) - u0)
@printf "ℓ = %.10f, closure |σ₋ℓ u(T) - u(0)| = %.1e, %d periods forward, converged %s\n" ℓ closure rpo.periods rpo.converged
expected = [real(ab8.exponents[1]), 0.0, 0.0, -2μ, real(ab8.exponents[3])]
println("| Exponent | Relative periodic orbit | Expected |")
for i = 1:5
    @printf "| λ%d | %+.12f (%+.1e) | %+.12f |\n" i real(rpo.exponents[i]) real(rpo.exponents[i]) expected[i]
end
@printf "largest difference from the expected exponents %.1e\n" maximum(abs, rpo.exponents .- expected)
@printf "imaginary parts at most %.1e, sum rule %.1e, |adjoint - forward| %.1e\n" maximum(abs ∘ imag, rpo.exponents) sum(real, rpo.exponents) + 41/3 + 2μ maximum(abs, rpo.adjointexponents .- rpo.exponents)
offspan(v, B) = (Q = Matrix(qr(B).Q); norm(v - Q*(Q'*v))/norm(v))
wf = 0.0
wg = 0.0
wr = 0.0
for k in eachindex(rpo.times)
    v = store(zeros(5), rpo.times[k])
    P = hcat(rpo.floquetvectors[k][rpo.order[2]], rpo.floquetvectors[k][rpo.order[3]])
    global wf = max(wf, offspan(lorenzphase(0.0, v, zeros(5)), P))
    global wg = max(wg, offspan([0, 0, 0, -v[5], v[4]], P))
    global wr = max(wr, offspan(rpo.floquetvectors[k][rpo.order[4]], reshape([0, 0, 0, v[4], v[5]], :, 1)))
end
@printf "f off the neutral plane %.1e, g off it %.1e, radial vector off (0, 0, 0, p, q) %.1e\n" wf wg wr
@printf "max |⟨ψᵢ, φⱼ⟩ - δᵢⱼ| %.1e\n" maximum(maximum(abs, [dot(ψ, φ) for ψ in rpo.adjointvectors[k], φ in rpo.floquetvectors[k]] - I) for k in eachindex(rpo.times))


# ============================================================================ #
# Figure 4: the relative periodic orbit over three periods                     #
# ============================================================================ #

three = RAMStorage(zeros(5); degree=5)
F(copy(u0), (0, 3T), three)
ts = times(three)
w  = [s[3]*complex(s[4], s[5]) for s in samples(three)]
fig = Figure(size=(560, 480))
ax  = Axis(fig[1, 1]; xlabel=L"\mathrm{Re}\,(z\,a)", ylabel=L"\mathrm{Im}\,(z\,a)", aspect=DataAspect())
for p = 1:3
    idx = findall(t -> (p - 1)*T - 1e-9 ≤ t ≤ p*T + 1e-9, ts)
    lines!(ax, real.(w[idx]), imag.(w[idx]); color=SLOTS[p], label="period $p")
end
starts = [w[findmin(abs.(ts .- p*T))[2]] for p = 0:3]
for s in starts
    lines!(ax, [0, real(s)], [0, imag(s)]; color=MUTED, linewidth=1)
end
scatter!(ax, real.(starts), imag.(starts); color=INK, markersize=10, strokecolor=:white, strokewidth=2)
scatter!(ax, [0.0], [0.0]; color=MUTED, markersize=6)
arc = [7.0*cis(θ) for θ in range(0, ℓ; length=40)]
lines!(ax, real.(arc), imag.(arc); color=INK2, linewidth=1.5)
text!(ax, 10.5*cos(ℓ/2), 10.5*sin(ℓ/2); text=L"\ell", align=(:center, :center), color=INK, fontsize=17)
axislegend(ax; position=:lb)
save(joinpath(ASSETS, "rpo.png"), fig; px_per_unit=2)
@printf "period starts at angles %s degrees, ℓ = %.2f degrees\n" join([@sprintf("%.2f", rad2deg(Base.angle(s))) for s in starts], ", ") rad2deg(ℓ)
println("\nwritten assets/convergence.png, assets/stretching.png, assets/vectors.png and assets/rpo.png")
