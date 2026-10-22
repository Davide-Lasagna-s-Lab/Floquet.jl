# The Floquet exponents and vectors of a relative periodic orbit of two-dimensional Kolmogorov
# flow at Re = 40, of period 60.80, which visits a burst, the example of the README. Makes
# assets/kolmogorov_exponents.png, assets/kolmogorov_vectors.png and
# assets/kolmogorov_vectors.gif, and prints every number that the README quotes about this example.
#
#     julia -t 4 --project=examples examples/kolmogorov.jl
#
# The vorticity of the orbit at t = 0 is read from data/kolmogorov_rpo.txt. The equations are
# those of OpenKolmogorovFlow.jl, and the schemes the IMEX schemes of Flows.jl. The run takes
# about fifteen minutes on a laptop, with four threads for the Fourier transforms.

using DelimitedFiles
using LinearAlgebra
using Printf
using Random
using FFTW
using Flows
using Floquet
using OpenKolmogorovFlow
using CairoMakie
using CairoMakie.Makie: FFMPEG_jll

include(joinpath(@__DIR__, "style.jl"))

const ASSETS = joinpath(@__DIR__, "..", "assets")


# ============================================================================ #
# The worked example of the README                                             #
# ============================================================================ #

t₀ = time()

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

# explicit and implicit parts of the linearised equations, and explicit part of the adjoint ones
lex, lim = splitexim(LinearisedEquation(n, m, Re, TangentMode(), FFTW.MEASURE))
aex, _   = splitexim(LinearisedEquation(n, m, Re, AdjointMode(), FFTW.MEASURE))

# eight random initial vectors, real fields of zero mean on the active modes
rng  = MersenneTwister(1)
init = [FFT(Field(randn(rng, 2m + 2, 2m + 2)), n) for j = 1:8]

# eight exponents and their vectors, in steps of about 0.02, with the shift back by ℓ along x
out = floquet(lex, 8, T/3040, store; A=lim, method=:CB4R3R4, init=init, oneevery=30,
              shift    = v -> xshift!(v, ℓ),
              shiftadj = v -> xshift!(v, -ℓ),
              vectors  = :floquet,
              Ladj     = aex)

seconds = time() - t₀
println("julia> round.(out.exponents; digits=6)")
show(stdout, MIME"text/plain"(), round.(out.exponents; digits=6))
println("\n\njulia> out.converged, out.periods")
show(stdout, MIME"text/plain"(), (out.converged, out.periods))
println("\n")


# ============================================================================ #
# Checks                                                                       #
# ============================================================================ #

# ---- the cost, the closure of the orbit and the exponents with all their digits ----
ΩT = samples(store)[end]
@printf "%.1f minutes with %d threads, %d periods of each sweep\n" seconds/60 Threads.nthreads() out.periods
@printf "closure |σ₋ℓ ω(T) - ω(0)|/|ω(0)| = %.1e, %d samples\n" sqrt(normdiff(xshift!(copy(ΩT), ℓ), Ω0))/norm(Ω0) length(times(store))
@printf "π/T = %.12f\n" π/T
for (i, λ) in enumerate(out.exponents)
    @printf "λ%d = %+.12f %+.12fi, adjoint %+.12f %+.12fi\n" i real(λ) imag(λ) real(out.adjointexponents[i]) imag(out.adjointexponents[i])
end
@printf "max |adjoint - forward| %.1e, order %s, pairs %s\n" maximum(abs, out.adjointexponents .- out.exponents) string(out.order) string(out.pairs)
@printf "sum of the unstable exponents %.6f\n" sum(real(λ) for λ in out.exponents if real(λ) > 1e-3)

# ---- against the Arnoldi iteration on the monodromy matrix of the shooting map ----
# CB3R2R3e with steps of 0.01, a Krylov space of dimension 60, from the merged catalogue
arnoldi = [0.123718953875 + 0.0im,
           0.090910845698 + 0.051667552682im,
           0.072825744419 + 0.051667552682im,
           0.024063996382 + 0.0im,
           0.0 + 0.0im,
           0.0 + 0.0im,
           -0.051216417279 + 0.036419174236im,
           -0.051216417279 - 0.036419174236im]
@printf "largest difference from the Arnoldi iteration %.1e\n" maximum(abs, out.exponents .- arnoldi)

# ---- the rates of convergence, the ratios of the moduli of consecutive multipliers ----
λs = [real.(out.exponents); -0.073092658290]
@printf "rates per period: %s\n" join([@sprintf("%.3f", exp((λs[j+1] - λs[j])*T)) for j = 1:8], " ")

# ---- the neutral plane, the biorthogonality, the dissipation ----
# the distance of v from the plane of a and b, relative to |v|, in the inner product of the fields
function offplane(v, a, b)
    q1 = copy(a)
    q1 ./= norm(q1)
    q2 = copy(b)
    q2 .-= dot(q1, q2) .* q1
    q2 ./= norm(q2)
    r  = copy(v)
    r .-= dot(q1, v) .* q1
    r .-= dot(q2, v) .* q2
    return norm(r)/norm(v)
end
feq = ForwardEquation(n, m, Re, kf, FFTW.MEASURE)
wf  = 0.0
wg  = 0.0
bi  = 0.0
for k in eachindex(out.times)
    u = store(similar(Ω0), out.times[k])
    Φ = out.floquetvectors[k]
    Ψ = out.adjointvectors[k]
    global wf = max(wf, offplane(feq(0.0, u, similar(u)), Φ[out.order[5]], Φ[out.order[6]]))
    global wg = max(wg, offplane(ddx!(similar(u), u), Φ[out.order[5]], Φ[out.order[6]]))
    global bi = max(bi, maximum(abs, [dot(ψ, φ) for ψ in Ψ, φ in Φ] - I))
end
@printf "f off the neutral plane %.1e, ∂ω/∂x off it %.1e, max |⟨ψᵢ, φⱼ⟩ - δᵢⱼ| %.1e\n" wf wg bi
Dlam = Re/(2kf^2)
ts   = times(store)
D    = [dissrate(x, Re)/Dlam for x in samples(store)]
tb   = ts[argmax(D)]
@printf "dissipation from %.4f to %.4f of the laminar value, largest at t = %.2f\n" minimum(D) maximum(D) tb


# ============================================================================ #
# Figure: the exponents and their convergence                                  #
# ============================================================================ #

# the colour of an exponent, unstable, neutral or stable, and its marker, real, negative
# multiplier or complex pair. The zero exponents are those of the neutral plane, whatever the
# small imaginary part that the iteration gives them.
role(λ)   = abs(real(λ)) < 1e-6 ? NEUTRAL : real(λ) > 0 ? UNSTABLE : STABLE
marker(λ) = abs(real(λ)) < 1e-6 || imag(λ) == 0 ? :circle :
            abs(imag(λ) - π/T) < 1e-9 ? :utriangle : :diamond
ring      = (marker=:circle, markersize=21, color=:transparent, strokecolor=INK, strokewidth=1.2)

λ   = out.exponents
fig = Figure(size=(820, 360))
ax1 = Axis(fig[1, 1]; xlabel=L"j", ylabel=L"\mathrm{Re}\,\lambda_j", xticks=1:8)
ax2 = Axis(fig[1, 2]; xlabel="period", ylabel=L"|\lambda_j^{(p)} - \lambda_j|", yscale=log10,
           yticks=LogTicks(-12:3:0), xticks=0:3:18)
hlines!(ax1, [0.0]; color=AXIS, linewidth=1)
scatter!(ax1, 1:8, real.(arnoldi); ring...)
for j in eachindex(λ)
    scatter!(ax1, [j], [real(λ[j])]; color=role(λ[j]), marker=marker(λ[j]), markersize=13,
             strokecolor=:white, strokewidth=1.5)
end
Legend(fig[0, 1], [MarkerElement(color=INK2, marker=:circle, markersize=11),
                   MarkerElement(color=INK2, marker=:utriangle, markersize=11),
                   MarkerElement(color=INK2, marker=:diamond, markersize=11),
                   MarkerElement(; ring..., markersize=15)],
       ["real", "negative multiplier", "complex pair", "Arnoldi iteration"];
       orientation=:horizontal, framevisible=false, labelcolor=INK2, tellwidth=false, nbanks=2)

# the distance of each exponent from its final value over the periods, one line per pair
P = length(out.history)
for j in eachindex(λ)
    marker(λ[j]) === :diamond && imag(λ[j]) < 0 && continue
    e = [max(abs(h[j] - λ[j]), 1e-15) for h in out.history[1:P-1]]
    scatterlines!(ax2, 1:P-1, e; color=role(λ[j]), markersize=7, strokecolor=:white, strokewidth=1)
end
ylims!(ax1, -0.075, 0.16)
ylims!(ax2, 1e-13, 1)
axislegend(ax2, [LineElement(color=c, linewidth=2) for c in (UNSTABLE, NEUTRAL, STABLE)],
           ["unstable", "neutral", "stable"]; position=:rt)
panel!(ax1, "(a)")
panel!(ax2, "(b)")
save(joinpath(ASSETS, "kolmogorov_exponents.png"), fig; px_per_unit=2)
println("written assets/kolmogorov_exponents.png")


# ============================================================================ #
# The vectors along the period                                                 #
# ============================================================================ #

# The Floquet vectors at t = 0 carried along the orbit by the linearised equations, with the
# same scheme and step, times exp(-Re λ t): the periodic factor of Floquet's theorem, up to the
# shift and the sign at T, whose size shows the growth along the orbit relative to the mean.
G  = flow(lex, lim, CB4R3R4(similar(Ω0), ContinuousMode(false)), TimeStepFromStorage(T/3040))
js = out.order[1:3]
P3 = map(1:3) do i
    mon = Monitor(similar(Ω0), (t, v) -> copy(v); oneevery=30)
    G(copy(out.floquetvectors[1][js[i]]), store, (0.0, T), mon)
    times(mon) ≈ out.times || error("the times of the monitor are not those of the vectors")
    map(zip(times(mon), samples(mon))) do (t, v)
        w = copy(v)
        w .*= exp(-real(out.exponents[i])*t)
        w
    end
end
for i = 1:3
    cs = [dot(P3[i][k], out.floquetvectors[k][js[i]])/norm(P3[i][k]) for k in eachindex(out.times)]
    s  = [norm(v) for v in P3[i]]
    a  = argmin(s)
    b  = argmax(s)
    @printf "φ%d carried along: largest 1 - |cos| with the Floquet vector %.1e, size from %.3f at t = %.1f to %.3f at t = %.1f\n" i maximum(1 .- abs.(cs)) s[a] out.times[a] s[b] out.times[b]
end

# the fields on the grid and on its periodic copy at 2π, in the frame that drifts with the orbit,
# x along the first index
function grid(U, t)
    V = xshift!(copy(U), ℓ*t/T)
    A = permutedims(parent(IFFT(V)))
    return A[[1:end; 1], [1:end; 1]]
end
xs = range(0, 2π; length=2m + 3)
K  = length(out.times)
Ωs = [grid(store(similar(Ω0), t), t) for t in out.times]
Vs = [[grid(P3[i][k], out.times[k]) for k = 1:K] for i = 1:3]
@printf "largest |ω| along the orbit %.2f\n" maximum(maximum(abs, A) for A in Ωs)

# ---- the layout of the still figure and of the animation, with each field scaled by s ----
names  = (L"\omega", L"\varphi_1", L"\varphi_2", L"\varphi_3")
tags   = ("(a)", "(b)", "(c)", "(d)")
fields = (Ωs, Vs...)
function layout(fig, k, s)
    obs = [Observable(fields[i][k] ./ s[i]) for i = 1:4]
    for i = 1:4
        ax = Axis(fig[1, i]; xlabel=L"x", ylabel=i == 1 ? L"y" : "", title=names[i],
                  aspect=DataAspect(), limits=(0, 2π, 0, 2π), xticks=([0, π, 2π], ["0", "π", "2π"]),
                  yticks=([0, π, 2π], ["0", "π", "2π"]), yticklabelsvisible=i == 1)
        heatmap!(ax, xs, xs, obs[i]; colormap=DIVERGING, colorrange=(-1, 1))
        hidespines!(ax)
        ax.xgridvisible = false
        ax.ygridvisible = false
        Label(fig[1, i, TopLeft()], tags[i]; font=:bold, color=INK, fontsize=15,
              padding=(0, 0, 4, 0), halign=:left)
    end
    Colorbar(fig[1, 5]; colormap=DIVERGING, limits=(-1, 1), ticks=[-1, 0, 1], width=10,
             labelcolor=INK2, ticklabelcolor=INK2, tickcolor=AXIS, spinewidth=0)
    return obs
end

# ---- the still figure, at the time closest to the peak of the dissipation, each field scaled
#      by its largest absolute value at this time ----
kb  = argmin(abs.(out.times .- tb))
fig = Figure(size=(820, 250))
layout(fig, kb, [maximum(abs, f[kb]) for f in fields])
resize_to_layout!(fig)
save(joinpath(ASSETS, "kolmogorov_vectors.png"), fig; px_per_unit=2)
@printf "still figure at t = %.2f, the peak of the burst at t = %.2f\n" out.times[kb] tb

# ---- the animation over one period, each field scaled by its largest absolute value over the
#      period. The frames are written as images, and made into a GIF by FFmpeg with a palette
#      of 96 colours, without dithering, which keeps the file small. The last frame, at T, is
#      the first one in the drifting frame, up to the sign of the vectors, and it is left out. ----
fig = Figure(size=(820, 270))
obs = layout(fig, 1, [maximum(maximum(abs, A) for A in f) for f in fields])
lab = Observable("")
Label(fig[0, 1:5], lab; color=INK2, fontsize=14, halign=:left, tellwidth=false)
resize_to_layout!(fig)
dir = mktempdir()
for k = 1:K-1
    for i = 1:4
        obs[i][] = fields[i][k] ./ maximum(maximum(abs, A) for A in fields[i])
    end
    lab[] = @sprintf("t = %.1f, dissipation %.3f of the laminar value", out.times[k],
                     dissrate(store(similar(Ω0), out.times[k]), Re)/Dlam)
    save(joinpath(dir, @sprintf("frame_%03d.png", k)), fig; px_per_unit=1)
end
gif     = joinpath(ASSETS, "kolmogorov_vectors.gif")
palette = "split[s0][s1];[s0]palettegen=max_colors=96:stats_mode=full[p];[s1][p]paletteuse=dither=none"
run(`$(FFMPEG_jll.ffmpeg()) -y -loglevel error -framerate 10 -i $(joinpath(dir, "frame_%03d.png")) -vf $palette -loop 0 $gif`)
@printf "written assets/kolmogorov_vectors.png and assets/kolmogorov_vectors.gif, %d frames, %.1f MB\n" K-1 filesize(gif)/1e6
@printf "%.1f minutes in all\n" (time() - t₀)/60
