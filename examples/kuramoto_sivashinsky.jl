# The Floquet exponents and vectors of a relative periodic orbit of the Kuramoto-Sivashinsky
# equation on a domain of length 22, the example of the README. Makes assets/ks_vectors.png and
# prints every number that the README quotes about this example.
#
#     julia --project=examples examples/kuramoto_sivashinsky.jl
#
# The orbit at t = 0 is read from data/ks_rpo.txt. The equations are those of KS.jl, with the
# products dealiased by padding, and the schemes are those of Flows.jl, with RK8.

using DelimitedFiles
using LinearAlgebra
using Printf
using Flows
using Floquet
using KS
using CairoMakie

include(joinpath(@__DIR__, "style.jl"))

const ASSETS = joinpath(@__DIR__, "..", "assets")


# ============================================================================ #
# The worked example of the README                                             #
# ============================================================================ #

# the domain [0, L), mapped on the [0, 2π) of KS.jl, with ν = (2π/L)² and n Fourier modes
L = 22
q = 2π/L
ν = q^2
n = 16

# right-hand side, linearised and adjoint equations, with the products dealiased by padding
f  = ForwardEquation(n, ν, false; dealias=:pad)
J  = LinearisedEquation(n, ν, false, TangentMode(); dealias=:pad)
Jᵀ = LinearisedEquation(n, ν, false, AdjointMode(); dealias=:pad)

# the same, times ν, in the time of the equation on [0, L)
ks(t, u, dudt)        = (f(t, u, dudt); dudt .*= ν)
ks_tan(t, u, v, dvdt) = (J(t, u, v, dvdt); dvdt .*= ν)
ks_adj(t, u, w, dwdt) = (Jᵀ(t, u, w, dwdt); dwdt .*= ν)

# the orbit at t = 0, its period and its shift, u(x, T) = u(x - ℓ, 0)
u0 = FTField(vec(readdlm(joinpath(@__DIR__, "data", "ks_rpo.txt"); comments=true)), false)
T  = 32.79688563412746
ℓ  = 10.958264212631029

# one period of the orbit, in K steps of RK8, in a storage with both ends
K     = 6560
store = RAMStorage(u0; degree=5)
F     = flow(ks, RK8(similar(u0)), TimeStepConstant(T/K))
uT    = F(copy(u0), (0, T), store)

# exponents and vectors, with the shift back by ℓ and its adjoint, on the [0, 2π) of KS.jl
out = floquet(ks_tan, 12, T/K, store; method=:RK8, λtol=1e-10, oneevery=10, maxiter=100,
              shift    = v -> shift!(v, q*ℓ),
              shiftadj = v -> shift!(v, -q*ℓ),
              vectors  = :floquet,
              Ladj     = ks_adj)

println("julia> round.(out.exponents; digits=6)")
show(stdout, MIME"text/plain"(), round.(out.exponents; digits=6))
println("\n\njulia> out.converged, out.periods")
show(stdout, MIME"text/plain"(), (out.converged, out.periods))
println("\n")


# ============================================================================ #
# Checks                                                                       #
# ============================================================================ #

# ---- the closure of the orbit, and the exponents with all their digits ----
@printf "closure |σ₋ℓ u(T) - u(0)|/|u(0)| = %.1e\n" norm(shift!(copy(uT), q*ℓ) - u0)/norm(u0)
@printf "π/T = %.12f\n" π/T
for (i, λ) in enumerate(out.exponents)
    @printf "λ%-2d = %+.12f %+.12fi, adjoint %+.12f %+.12fi\n" i real(λ) imag(λ) real(out.adjointexponents[i]) imag(out.adjointexponents[i])
end
@printf "max |adjoint - forward| %.1e, order %s, pairs %s\n" maximum(abs, out.adjointexponents .- out.exponents) string(out.order) string(out.pairs)
@printf "zero exponents %.1e and %.1e, phase of the unstable pair over a period %.4f\n" real(out.exponents[3]) real(out.exponents[4]) imag(out.exponents[1])*T

# ---- the eigenvalues of the comoving monodromy matrix, from the identity ----
# The tangent of RK8 coupled to the orbit is the exact derivative of its map.
ks_tan(t, u, dudt, v, dvdt) = ks_tan(t, u, v, dvdt)
G = flow(couple(ks, ks_tan), RK8(couple(similar(u0), similar(u0))), TimeStepConstant(T/K))
M = zeros(2n, 2n)
for j = 1:2n
    z = couple(copy(u0), FTField(n, false))
    z[2][j] = 1
    G(z, (0, T))
    M[:, j] .= shift!(z[2], q*ℓ)
end
μ  = eigvals(M)
λM = sort(log.(complex(μ))./T; by=z -> (real(z), imag(z)), rev=true)
println("\n| j | floquet | eigenvalues of M | |Λⱼ/Λ₁| |")
for i = 1:12
    @printf "| %d | %+.6f | %+.6f | %.1e |\n" i real(out.exponents[i]) real(λM[i]) exp((real(out.exponents[i]) - real(out.exponents[1]))*T)
end
@printf "largest difference over the first eight %.1e\n" maximum(abs, out.exponents[1:8] .- λM[1:8])
@printf "real eigenvalues of M among the first eight: %s\n" join([@sprintf("%+.6f", real(v)) for v in μ if abs(imag(v)) < 1e-12 && abs(v) > 1e-6], " ")
@printf "rounding floor λ₁ - log(1/ε)/T = %.3f\n" real(out.exponents[1]) - log(1/eps())/T

# ---- the neutral plane, and the biorthogonality ----
offspan(v, P) = (Q = Matrix(qr(P).Q); norm(v - Q*(Q'*v))/norm(v))
wf = 0.0
wg = 0.0
bi = 0.0
for k in eachindex(out.times)
    u = store(similar(u0), out.times[k])
    Φ = out.floquetvectors[k]
    Ψ = out.adjointvectors[k]
    P = hcat(Vector(Φ[out.order[3]]), Vector(Φ[out.order[4]]))
    g = ddx!(similar(u), u)
    global wf = max(wf, offspan(Vector(ks(0.0, u, similar(u))), P))
    global wg = max(wg, offspan(Vector(g), P))
    global bi = max(bi, maximum(abs, [dot(ψ, φ) for ψ in Ψ, φ in Φ] - I))
end
@printf "f off the neutral plane %.1e, g off it %.1e, max |⟨ψᵢ, φⱼ⟩ - δᵢⱼ| %.1e\n" wf wg bi

# ---- the vectors at T, the vectors at 0 moved by σℓ and multiplied by the sign of their multiplier ----
for i in (5, 6)
    j  = out.order[i]
    φ0 = shift!(copy(out.floquetvectors[1][j]), -q*ℓ)
    φT = out.floquetvectors[end][j]
    @printf "λ%d: |φ(T) + σℓ φ(0)| = %.1e, |φ(T) - σℓ φ(0)| = %.1e\n" i norm(φT + φ0) norm(φT - φ0)
end

# ---- the continuity of the vectors of the pair, against that of the Schur vectors ----
jump(V, j) = maximum(k -> norm(V[k+1][j] - V[k][j]), 1:length(V)-1)
@printf "largest change between consecutive times: φ₁ %.1e, φ₂ %.1e, Schur vectors %.1e and %.1e\n" jump(out.floquetvectors, 1) jump(out.floquetvectors, 2) jump(out.bases, 1) jump(out.bases, 2)


# ============================================================================ #
# Figure: the exponents and their convergence                                  #
# ============================================================================ #

# the colour of an exponent, unstable, neutral or stable, and its marker, real, negative
# multiplier or complex pair
role(λ)   = abs(real(λ)) < 1e-6 ? NEUTRAL : real(λ) > 0 ? UNSTABLE : STABLE
marker(λ) = imag(λ) == 0 ? :circle : abs(imag(λ) - π/T) < 1e-9 ? :utriangle : :diamond
ring      = (marker=:circle, markersize=21, color=:transparent, strokecolor=INK, strokewidth=1.2)

λ   = out.exponents
fig = Figure(size=(820, 360))
ax1 = Axis(fig[1, 1]; xlabel=L"j", ylabel=L"\mathrm{Re}\,\lambda_j", xticks=1:12)
ax2 = Axis(fig[1, 2]; xlabel="period", ylabel=L"|\lambda_j^{(p)} - \lambda_j|", yscale=log10,
           yticks=LogTicks(-15:3:0))
rounding = real(λ[1]) - log(1/eps())/T
hlines!(ax1, [0.0]; color=AXIS, linewidth=1)
hlines!(ax1, [rounding]; color=MUTED, linewidth=1, linestyle=:dash)
text!(ax1, 0.6, rounding; text="rounding", align=(:left, :bottom), color=INK2, fontsize=12)
scatter!(ax1, 1:12, real.(λM[1:12]); ring...)
for j in eachindex(λ)
    scatter!(ax1, [j], [real(λ[j])]; color=role(λ[j]), marker=marker(λ[j]), markersize=13,
             strokecolor=:white, strokewidth=1.5)
end
Legend(fig[0, 1], [MarkerElement(color=INK2, marker=:circle, markersize=11),
                   MarkerElement(color=INK2, marker=:utriangle, markersize=11),
                   MarkerElement(color=INK2, marker=:diamond, markersize=11),
                   MarkerElement(; ring..., markersize=15)],
       ["real", "negative multiplier", "complex pair", L"\text{eigenvalues of } \tilde{M}"];
       orientation=:horizontal, framevisible=false, labelcolor=INK2, tellwidth=false, nbanks=2)

# the distance of each exponent from its final value over the periods, one line per pair
P = length(out.history)
for j in eachindex(λ)
    (imag(λ[j]) < 0 && abs(imag(λ[j]) - π/T) > 1e-9) && continue
    e = [max(abs(h[j] - λ[j]), 1e-16) for h in out.history[1:P-1]]
    lines!(ax2, 1:P-1, e; color=role(λ[j]), linewidth=1.5)
end
ylims!(ax1, -6.2, 1.2)
ylims!(ax2, 1e-16, 10)
axislegend(ax2, [LineElement(color=c, linewidth=2) for c in (UNSTABLE, NEUTRAL, STABLE)],
           ["unstable", "neutral", "stable"]; position=:rt)
panel!(ax1, "(a)")
panel!(ax2, "(b)")
save(joinpath(ASSETS, "ks_exponents.png"), fig; px_per_unit=2)
println("written assets/ks_exponents.png")


# ============================================================================ #
# Figure: the orbit and two Floquet vectors in space and time                  #
# ============================================================================ #

# a state at the points x of [0, L): u = 2 Re Σₖ Uₖ exp(ikqx), times q for the field on [0, L)
field(U, xs) = [2q*real(sum(U[WaveNumber(k)]*cis(k*q*x) for k = 1:n)) for x in xs]

xs  = range(0, L; length=177)
ts  = out.times
uxt = reduce(hcat, [field(store(similar(u0), t), xs) for t in ts])
φ1  = reduce(hcat, [field(out.floquetvectors[k][out.order[1]], xs) for k in eachindex(ts)])
φ5  = reduce(hcat, [field(out.floquetvectors[k][out.order[5]], xs) for k in eachindex(ts)])
@printf "\nlargest |u| along the orbit %.2f\n" maximum(abs, uxt)

fig  = Figure(size=(820, 440))
data = ((uxt, L"u", "(a)"), (φ1, L"\varphi_1", "(b)"), (φ5, L"\varphi_5", "(c)"))
for (i, (A, name, tag)) in enumerate(data)
    ax = Axis(fig[1, i]; xlabel=L"x", ylabel=i == 1 ? L"t" : "", title=name, limits=(0, L, 0, T),
              xticks=0:11:22, yticks=0:8:32, yticklabelsvisible=i == 1)
    heatmap!(ax, xs, ts, A ./ maximum(abs, A); colormap=DIVERGING, colorrange=(-1, 1))
    hidespines!(ax)
    ax.xgridvisible = false
    ax.ygridvisible = false
    Label(fig[1, i, TopLeft()], tag; font=:bold, color=INK, fontsize=15, padding=(0, 0, 4, 0),
          halign=:left)
end
Colorbar(fig[1, 4]; colormap=DIVERGING, limits=(-1, 1), ticks=[-1, 0, 1], width=12,
         label="scaled by the largest value", labelcolor=INK2, ticklabelcolor=INK2,
         tickcolor=AXIS, spinewidth=0)
save(joinpath(ASSETS, "ks_vectors.png"), fig; px_per_unit=2)
println("written assets/ks_vectors.png")
