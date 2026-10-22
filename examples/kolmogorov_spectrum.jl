# Fifty Floquet exponents of the relative periodic orbit of Kolmogorov flow of period 60.80, from
# the periodic Schur decomposition of Floquet.jl, against the eigenvalues of the monodromy matrix
# from the Arnoldi iteration, with the same integration of the linearised equations. Makes
# assets/kolmogorov_exponents.png and prints every number that the README quotes about it.
#
#     julia -t 4 --project=examples examples/kolmogorov_spectrum.jl
#
# The orbit and the equations are those of kolmogorov.jl. The run takes about two hours on a
# laptop, with four threads for the Fourier transforms, most of it for the 36 periods of floquet.

using DelimitedFiles
using LinearAlgebra
using Printf
using Random
using FFTW
using Flows
using Floquet
using OpenKolmogorovFlow
using CairoMakie

include(joinpath(@__DIR__, "style.jl"))

const ASSETS = joinpath(@__DIR__, "..", "assets")

t₀ = time()


# ============================================================================ #
# The orbit and the linearised equations, as in kolmogorov.jl                  #
# ============================================================================ #

# Re = 40 and forcing wavenumber 4, wavenumbers up to n = 33, dealiased on the grid of m = 49
n  = 33
m  = 49
Re = 40
kf = 4

# the vorticity of the orbit at t = 0, its period and its shift, ω(x, y, T) = ω(x - ℓ, y, 0)
c  = readdlm(joinpath(@__DIR__, "data", "kolmogorov_rpo.txt"); comments=true)
Ω0 = FTField(n, m)
for (kx, ky, a, b) in eachrow(c)
    Ω0[WaveNumber(Int(ky), Int(kx))] = complex(a, b)
end
T = 60.80397639342852
ℓ = -2.9035006731854844

# one period of the orbit, over the five segments of the shooting, with its scheme and its step,
# in a storage with both ends
fex, fim = splitexim(ForwardEquation(n, m, Re, kf, FFTW.MEASURE))
F        = flow(fex, fim, CB3R2R3e(FTField(n, m)), TimeStepConstant(0.01))
store    = RAMStorage(FTField(n, m); storelast=false, degree=5)
Ω        = copy(Ω0)
for k = 1:5
    F(Ω, ((k - 1)*T/5, k*T/5), store)
end
push!(store, T, copy(Ω))

# explicit and implicit parts of the linearised equations
lex, lim = splitexim(LinearisedEquation(n, m, Re, TangentMode(), FFTW.MEASURE))


# ============================================================================ #
# Fifty exponents by the periodic Schur decomposition                          #
# ============================================================================ #

# fifty random initial vectors, real fields of zero mean on the active modes
rng  = MersenneTwister(1)
init = [FFT(Field(randn(rng, 2m + 2, 2m + 2)), n) for j = 1:50]

# fifty exponents without the vectors, over 36 periods
deep = floquet(lex, 50, T/3040, store; A=lim, method=:CB4R3R4, init=init, oneevery=50,
               shift = v -> xshift!(v, ℓ), λtol=1e-10, maxiter=36)

t₁ = time()
@printf "floquet: %d periods, converged %s, %.1f minutes with %d threads\n" deep.periods deep.converged (t₁ - t₀)/60 Threads.nthreads()


# ============================================================================ #
# The eigenvalues of the monodromy matrix, by the Arnoldi iteration            #
# ============================================================================ #

# the monodromy matrix in the frame of the orbit, M̃ v = σ₋ℓ Y(T, 0) v, applied by one integration
# of the linearised equations over the period, with the scheme and the step of floquet
G = flow(lex, lim, CB4R3R4(similar(Ω0), ContinuousMode(false)), TimeStepFromStorage(T/3040))
function monodromy!(v)
    G(v, store, (0.0, T))
    return xshift!(v, ℓ)
end

# K steps of the Arnoldi iteration from a random field, with Gram-Schmidt twice in each step
K = 100
Q = [FFT(Field(randn(MersenneTwister(2), 2m + 2, 2m + 2)), n)]
Q[1] ./= norm(Q[1])
H = zeros(K + 1, K)
for j = 1:K
    w = monodromy!(copy(Q[j]))
    for _ = 1:2, i = 1:j
        h = dot(Q[i], w)
        H[i, j] += h
        w .-= h .* Q[i]
    end
    H[j+1, j] = norm(w)
    w ./= H[j+1, j]
    push!(Q, w)
end

# the exponents of the Ritz values, sorted by real part, and the residual of each Ritz pair
E  = eigen(H[1:K, 1:K])
λA = log.(complex(E.values)) ./ T
p  = sortperm(λA; by=z -> (real(z), imag(z)), rev=true)
λA = λA[p]
ρA = abs.(H[K+1, K] .* E.vectors[K, p]) ./ abs.(E.values[p])

t₂ = time()
@printf "Arnoldi: %d steps, %.1f minutes, last subdiagonal entry %.1e\n" K (t₂ - t₁)/60 H[K+1, K]


# ============================================================================ #
# Comparison and convergence                                                   #
# ============================================================================ #

λ  = deep.exponents
N  = length(λ)
fl = real(λ[1]) - log(1/eps())/T
P  = deep.periods
dλ = abs.(deep.history[P] .- deep.history[P-1])
println("\n|  j | floquet               | Arnoldi               | |Λⱼ/Λ₁|  | change over the last period |")
for j = 1:N
    @printf "| %2d | %+.6f %+.6fi | %+.6f %+.6fi | %.1e | %.1e |\n" j real(λ[j]) imag(λ[j]) real(λA[j]) imag(λA[j]) exp((real(λ[j]) - real(λ[1]))*T) dλ[j]
end

# ---- where the eigenvalues of the monodromy matrix depart from the exponents of floquet ----
j0 = findfirst(j -> abs(real(λA[j] - λ[j])) > 1e-10, 1:N) - 1
@printf "\nthe first %d exponents agree to %.1e\n" j0 maximum(abs, λA[1:j0] .- λ[1:j0])
@printf "at j = 22: floquet %.6f, Arnoldi %.6f, difference %.1e, |Λ₂₂/Λ₁| = %.1e\n" real(λ[22]) real(λA[22]) abs(real(λA[22] - λ[22])) exp((real(λ[22]) - real(λ[1]))*T)
@printf "rounding level λ₁ - log(1/ε)/T = %.3f, Arnoldi from j = 33 to %d between %.3f and %.3f\n" fl K minimum(real, λA[33:K]) maximum(real, λA[33:K])
@printf "floquet: %d exponents below the rounding level, the deepest %.3f\n" count(z -> real(z) < fl, λ) real(λ[N])

# ---- the convergence over the periods, and the groups of exponents not yet separated ----
for tol in (1e-10, 1e-8, 1e-6, 1e-4)
    @printf "exponents that changed by less than %.0e over the last period: %d of %d\n" tol count(<(tol), dλ) N
end
slow   = findall(>(1e-4), dλ)
groups = [[slow[1]]]
for j in slow[2:end]
    j - groups[end][end] ≤ 2 ? push!(groups[end], j) : push!(groups, [j])
end
for g in groups
    s(p) = sum(real(deep.history[p][i]) for i in g)
    @printf "not separated: exponents %s near %.3f, the sum of their real parts %.6f changed by %.1e over the last period\n" string(g) s(P)/length(g) s(P) abs(s(P) - s(P-1))
end
@printf "%.1f minutes in all\n" (time() - t₀)/60


# ============================================================================ #
# Figure: the exponents and their convergence                                  #
# ============================================================================ #

# the colour of an exponent, unstable, neutral, stable, or stable below the rounding level of the
# Arnoldi iteration, and its marker, real, negative multiplier or complex pair. The zero
# exponents are those of the neutral plane, whatever small imaginary part they get.
const DEEP = "#0d366b"
role(z)   = abs(real(z)) < 1e-6 ? NEUTRAL : real(z) > 0 ? UNSTABLE : real(z) > fl ? STABLE : DEEP
marker(z) = abs(real(z)) < 1e-6 || imag(z) == 0 ? :circle :
            abs(imag(z) - π/T) < 1e-9 ? :utriangle : :diamond
ring      = (marker=:circle, markersize=15, color=:transparent, strokecolor=INK, strokewidth=1)

fig = Figure(size=(820, 380))
ax1 = Axis(fig[1, 1]; xlabel=L"j", ylabel=L"\mathrm{Re}\,\lambda_j", xticks=0:10:N)
ax2 = Axis(fig[1, 2]; xlabel="period", ylabel=L"j", yreversed=true, xticks=0:6:36,
           yticks=[1, 10, 20, 30, 40, 50], xgridvisible=false, ygridvisible=false)
hlines!(ax1, [0.0]; color=AXIS, linewidth=1)
hlines!(ax1, [fl]; color=MUTED, linewidth=1, linestyle=:dash)
text!(ax1, 1, fl; text="rounding", align=(:left, :top), offset=(0, -2), color=INK2, fontsize=12)
scatter!(ax1, 1:N, real.(λA[1:N]); ring...)
for j = 1:N
    # the exponents not yet separated from their neighbours are drawn hollow
    c = role(λ[j])
    scatter!(ax1, [j], [real(λ[j])]; marker=marker(λ[j]), markersize=9,
             color=j in slow ? :white : c, strokecolor=j in slow ? c : :white, strokewidth=1.2)
end
ylims!(ax1, -0.72, 0.3)
Legend(fig[0, 1], [MarkerElement(color=INK2, marker=:circle, markersize=10),
                   MarkerElement(color=INK2, marker=:utriangle, markersize=10),
                   MarkerElement(color=INK2, marker=:diamond, markersize=10),
                   MarkerElement(color=:white, marker=:circle, markersize=10, strokecolor=INK2,
                                 strokewidth=1.2),
                   MarkerElement(; ring..., markersize=13)],
       ["real", "negative multiplier", "complex pair", "not separated", "Arnoldi iteration"];
       orientation=:horizontal, framevisible=false, labelcolor=INK2, tellwidth=false, nbanks=2)

# the change of each exponent from one period to the next, on a logarithmic scale
C = [log10(max(abs(deep.history[p][j] - deep.history[p-1][j]), 1e-16)) for p = 2:P, j = 1:N]
hm = heatmap!(ax2, 2:P, 1:N, C; colormap=cgrad(["#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"]),
              colorrange=(-14, 0))
Colorbar(fig[1, 3], hm; label="change over a period", ticks=(-14:7:0, ["10⁻¹⁴", "10⁻⁷", "1"]),
         width=10, labelcolor=INK2, ticklabelcolor=INK2, tickcolor=AXIS, spinewidth=0)
panel!(ax1, "(a)")
text!(ax2, 0.02, 0.98; text="(b)", space=:relative, align=(:left, :top), font=:bold,
      color=:white, fontsize=15)
save(joinpath(ASSETS, "kolmogorov_exponents.png"), fig; px_per_unit=3)
println("written assets/kolmogorov_exponents.png")
