<p align="center">
  <img src="assets/logo.svg" alt="Lyapunov.jl" width="720">
</p>

# Lyapunov.jl

Lyapunov.jl computes the Floquet exponents of periodic orbits of dynamical
systems. It is designed for long periodic orbits of chaotic systems, possibly
of large dimension, where the monodromy matrix cannot be formed or
eigendecomposed directly. The package implements the matrix-free algorithm of
appendix A of Lasagna (2020), which follows the periodic eigendecomposition of
Ding and Cvitanović (2016). The orbit and the linearised equations are
integrated with [Flows.jl](https://github.com/Davide-Lasagna-s-Lab/Flows.jl).

## Floquet exponents

Let $x(t)$ be a periodic orbit of $\dot x = f(x)$, with period $T$. A small
perturbation $y$ of the orbit obeys the linearised equations
$\dot y = J(x(t))\, y$, where $J$ is the Jacobian of $f$. Over one period the
perturbation is mapped by the monodromy matrix $M$, with $y(T) = M y(0)$. The
eigenvalues $\Lambda_k$ of $M$ are the Floquet multipliers. The Floquet
exponents are $\lambda_k = \log \Lambda_k / T$. Their real parts are the mean
rates at which perturbations grow or decay along the orbit. Their imaginary
parts are defined modulo $2\pi/T$. The exponents with positive real part give
the unstable directions of the orbit. An orbit of an autonomous system also
has a zero exponent, in the direction of the flow.

## Comparison with the eigenvalues of the monodromy matrix

The usual approach computes the multipliers as the eigenvalues of $M$. For a
small system, $M$ is built by propagating the identity over one period. For a
large system, $M$ is not formed. The Arnoldi iteration is applied instead to
the action of the propagator over one period, at the cost of one integration
of the linearised equations per iteration. This gives accurate leading
multipliers. It loses accuracy for the subleading multipliers, and for the
contracting ones in particular, on long and strongly unstable orbits.

The reason is the spread of the multipliers. Over one period they are
$e^{\lambda_k T}$, and on a long orbit of a chaotic system they span many
orders of magnitude. A vector propagated over a whole period aligns with the
leading direction. Its components along the other directions shrink, relative
to the leading one, by the ratios $|\Lambda_k/\Lambda_1|$. Once a ratio falls
below the machine precision, about $10^{-16}$, that multiplier is lost to
rounding. The periodic Schur decomposition avoids this (Lust 2001, Ding and
Cvitanović 2016, Lasagna 2020, appendix A). It re-orthonormalises the vectors
along the orbit, before they align. Each exponent is then computed from
well-conditioned factors, and stays accurate on long and strongly unstable
orbits.

The Lorenz system gives a check of accuracy. Its divergence is constant, so
every periodic orbit satisfies
$\lambda_1 + \lambda_2 + \lambda_3 = -(\sigma + 1 + \beta) = -41/3$, with
$\lambda_2 = 0$. The table compares the stable exponent $\lambda_3$ from
`floquet` with the one from the eigenvalues of $M$. Here $M$ is built by
propagating the identity over one period, which is what the Arnoldi iteration
would converge to in three dimensions. Both use RK4 with a time step of about
$4\times 10^{-4}$. The last orbit makes ten loops around the two wings of the
attractor. The errors are those of the sum rule,
$|\lambda_1 + \lambda_2 + \lambda_3 + 41/3|$.

| Orbit    | Period $T$ | $\Lambda_3$           | $\lambda_3$, `floquet` | $\lambda_3$, eigenvalues of $M$ | Error, `floquet`    | Error, eigenvalues of $M$ |
|----------|------------|-----------------------|------------------------|---------------------------------|---------------------|---------------------------|
| AB       | 1.559      | $1.2\times 10^{-10}$  | −14.661317             | −14.661320                      | $2\times 10^{-10}$  | $3\times 10^{-6}$         |
| AAB      | 2.306      | $2.2\times 10^{-15}$  | −14.627436             | −14.010890                      | $2\times 10^{-10}$  | 0.62                      |
| 10 loops | 7.416      | $1.5\times 10^{-47}$  | −14.539990             | −3.967873                       | $2\times 10^{-10}$  | 10.6                      |

The eigenvalues of $M$ give $\Lambda_3$ with a relative error of about
$10^{-16}\,|\Lambda_1/\Lambda_3|$. On the orbit AB this ratio is about
$4\times 10^{10}$, which explains the error of $3\times 10^{-6}$. On the
orbit AAB the ratio is about $4\times 10^{15}$, and $\lambda_3$ is off by
0.62. On the orbit with ten loops, the stable multiplier is
about $10^{-47}$. The eigenvalues of $M$ return a value near $10^{-13}$, set by
rounding, and a stable exponent of −3.97 instead of −14.54. The leading
exponents of the two methods agree to better than $10^{-10}$ on all three
orbits.

## The algorithm

The package never forms $M$. It marches $N$ tangent vectors with the
linearised equations, along the base orbit interpolated from a storage. The
vectors start as the first $N$ canonical directions, or as vectors given by
the user. Left alone they would
all align with the leading direction. To keep them independent they are
re-orthonormalised every few time steps by two-pass Gram–Schmidt. This is a
QR decomposition of the matrix of tangent vectors. The orthonormal factor
replaces the vectors, and the upper triangular factor $R_j$ is stored. Over
one period this gives $M Q_0 = Q_1 R_m \cdots R_2 R_1$, where $Q_0$ and $Q_1$
hold the initial and final orthonormal vectors.

The next period starts from $Q_1$. This is subspace iteration applied to $M$.
Over the periods, the first $k$ vectors span the invariant subspace of the $k$
leading multipliers, for every $k$. Their error decays as
$|\Lambda_{k+1}/\Lambda_k|$ per period.

At the end of each period the package solves an orthogonal Procrustes
problem. It finds the orthogonal matrix $\tilde Q$ that best maps the final
vectors onto the initial ones, $Q_0 \approx Q_1 \tilde Q$, from the singular
value decomposition of $Q_1^\top Q_0$. Once the subspaces have converged,
$\tilde Q$ is block diagonal. A real multiplier gives a diagonal entry of
$\pm 1$. The entry is −1 when the multiplier is negative, because the vector
then flips at each period. A complex-conjugate pair gives a $2\times 2$
rotation block, because the two vectors keep rotating in the invariant plane
of the pair.

The multipliers follow from the triangular factors. A real exponent is the
sum of the logarithms of a diagonal entry of all the $R_j$, divided by $T$.
For a negative multiplier, π is added to the imaginary part. A complex pair
comes from the eigenvalues of $\tilde Q^\top$ times the product of the
$2\times 2$ diagonal blocks of the $R_j$. This product is renormalised after
each factor, and the scale is accumulated as a logarithm. Large and small
factors are thus never multiplied together, and the exponents keep their
accuracy however far apart they are. The same idea underlies the periodic
Schur decomposition used by Lust (2001) and by Ding and Cvitanović (2016).

The periods are repeated until the logarithms of the multipliers change by
less than a tolerance between two periods.

## Installation

Lyapunov.jl and Flows.jl are not registered. In the package mode of the Julia
REPL, add both from GitHub.

```julia
pkg> add https://github.com/Davide-Lasagna-s-Lab/Flows.jl
pkg> add https://github.com/Davide-Lasagna-s-Lab/Lyapunov.jl
```

Julia 1.11 or later is required.

## Usage

The package exports one function.

```julia
out = floquet(L, N, dt, store; kwargs...)
```

`L` is the linearised operator. It is a callable `L(t, u, y, dydt)` that
writes in `dydt` the Jacobian at the base state `u` applied to the tangent
vector `y`. The base state is interpolated from `store` at time `t`. The
function makes `N` deep copies of `L`, so `L` may own work buffers.

`N` is the number of tangent vectors, hence of exponents. `dt` is the time
step of the linearised equations.

`store` is a Flows.jl `RAMStorage` holding one period of the orbit. For a
periodic orbit, build it with `period=T` and `storelast=false`, and fill it
by integrating the orbit over `(0, T)`. It must not hold a sample at or near
`t = T`, which would duplicate the sample at `t = 0`. The period is read from
the storage.

The keyword arguments are listed below.

| Keyword    | Default    | Meaning |
|------------|------------|---------|
| `A`        | `nothing`  | Linear part of the linearised operator, treated implicitly. `L` is then the explicit part only. |
| `method`   | `:RK4`     | Scheme for the linearised equations. `:RK4` is explicit. `:CNRK2`, `:CB3R2R2`, `:CB3R2R3c`, `:CB3R2R3e` and `:CB4R3R4` are IMEX schemes, used with `A`. |
| `oneevery` | `1`        | Number of time steps between two re-orthonormalisations. |
| `shift`    | `identity` | Function applied in place to each tangent vector at the end of every period. See the section on relative periodic orbits. |
| `init`     | `nothing`  | Initial tangent vectors, `N` states orthonormalised before use. The default is the first `N` canonical directions. Pass `init` when these are not meaningful directions of the state, for instance Fourier fields with padding or symmetry constraints. |
| `λtol`     | `1e-7`     | Tolerance on the change of the log-multipliers between two periods. |
| `θtol`     | `1e-7`     | Tolerance to detect the $2\times 2$ rotation blocks of $\tilde Q$ that flag complex pairs. |
| `maxiter`  | `50`       | Maximum number of periods. |
| `verbose`  | `false`    | Print the exponents after each period. |

The result is a named tuple `(exponents, converged, periods, history)`.
`exponents` is a vector of `N` complex Floquet exponents, sorted by
decreasing real part. Imaginary parts lie in $[-\pi/T, \pi/T]$. A negative
real multiplier has imaginary part $\pi/T$. In a complex pair, the exponent
with positive imaginary part comes first. `converged` tells whether the
tolerance was met, and `periods` is the number of periods marched. `history`
holds the exponents after each period.

The IMEX schemes need the operator `A` to support the `mul!`, `ImcA!` and
`ImcA_mul!` interface of Flows.jl, which Flows.jl provides for `Diagonal`.

## Worked example on the Lorenz orbit AB

This example computes the exponents of the shortest periodic orbit of the
Lorenz system, AB in the notation of Viswanath (2003). The point `x0` and
the period `T` below were converged by Newton shooting, with the same RK4 time
step as used here. The orbit closes to $2\times 10^{-12}$. The test suite
contains the Newton shooting.

```julia
using Flows
using Lyapunov

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
```

The output is the following.

```julia
julia> out.exponents
3-element Vector{ComplexF64}:
     0.9946500250317319 + 0.0im
 -1.503433853392455e-12 + 0.0im
      -14.6613166914551 + 0.0im

julia> out.converged, out.periods
(true, 17)

julia> sum(real, out.exponents) + 41/3
2.417941402654833e-10
```

The leading exponent agrees with the one from the eigenvalues of the monodromy
matrix to $4\times 10^{-13}$. The second exponent is zero, as expected for the
direction of the flow. The sum of the exponents matches the trace of the
Jacobian, −41/3, to $2\times 10^{-10}$.

## Relative periodic orbits

Systems with a continuous symmetry, such as translations in a periodic domain,
have relative periodic orbits. Such an orbit returns to a shifted copy of its
initial state, $x(T) = g\, x(0)$, where $g$ is an element of the symmetry
group. Its Floquet multipliers are the eigenvalues of $g^{-1} M$.

The keyword `shift` handles this case. It is a function that applies $g^{-1}$
in place to a tangent vector, that is, the shift that maps $x(T)$ back onto
$x(0)$. It is applied to every tangent vector at the end of every period,
before the Procrustes step. The shift must preserve the inner product, as
translations do. Since $x(T) \neq x(0)$, the storage must not be periodic. It
must hold both endpoints.

```julia
# s is the drift of the orbit over one period, and translate! is user code
# that translates a field in place
shift_back!(q) = translate!(q, -s)

store = RAMStorage(x0; degree=5)
F(copy(x0), (0, T), store)

out = floquet(L, N, dt, store; shift=shift_back!)
```

A relative periodic orbit has one more zero exponent, in the direction of the
symmetry.

## Choosing the parameters

### Number of tangent vectors

The exponents converge from the first to the last. The $k$-th exponent
converges at the rate $|\Lambda_{k+1}/\Lambda_k|$ per period. The last one
depends on the first multiplier left out, $\Lambda_{N+1}$. Computing a few more
exponents than needed makes the last wanted ones converge faster.

`N` must not split a complex pair. If the $N$-th and $(N+1)$-th multipliers
form a pair, the subspace of the first $N$ vectors is not invariant and the
last exponent does not converge. Increasing `N` by one solves this.

The cost of one period is that of `N` integrations of the linearised
equations, plus one Gram–Schmidt step of cost $N^2$ times the number of
degrees of freedom per re-orthonormalisation.

### Time step and storage

The accuracy of the exponents is set by the integration of the linearised
equations and by the interpolation of the base orbit. It is not set by the
conditioning of $M$. A Lagrange interpolation of degree 5, with samples as
dense as the time steps of the orbit, keeps the interpolation error well
below that of the time integration. The explicit `:RK4` and the IMEX
`:CB4R3R4` are fourth-order schemes. `:CB3R2R3c` and `:CB3R2R3e` are third
order, `:CB3R2R2` and `:CNRK2` are second order.

A periodic storage filled with `TimeStepConstant(T/n)` and `storelast=false`
holds exactly `n` samples, the first at `t = 0` and the last at `t = T - T/n`.
`floquet` checks that a periodic storage holds no sample at `t = T`.

### Re-orthonormalisation interval

Between two re-orthonormalisations, the tangent vectors grow at different
rates and tend to align with the leading direction. The vectors must be
re-orthonormalised before the ratio between the growth of the first and of the
last vector comes near $10^{16}$, or the trailing vectors lose digits. As a
rule, `oneevery*dt` times the spread of the instantaneous growth rates should
stay below about 10. For the orbit AB above, the results do not change up to
`oneevery = 1000`. Each re-orthonormalisation stores an $N\times N$ triangular
matrix, so a larger `oneevery` also saves memory on long orbits.

## Convergence

The tolerance `λtol` applies to the logarithms of the multipliers. The
exponents change by `λtol/T` at convergence. The number of periods depends on
the gaps between the multipliers over one period,
$e^{(\lambda_{k+1} - \lambda_k) T}$. Long orbits need few periods. Short orbits may need many. The
orbit AB needs 17 periods to reach `λtol = 1e-10`, because the zero exponent
converges at the rate $e^{-0.99\,T} \approx 0.21$ per period. With the default
`λtol = 1e-7` it needs 13 periods, and the second exponent is
$-6\times 10^{-9}$. Check `converged`, and inspect
`history` or set `verbose = true` to follow the iteration.

```
period   1:  +1.1654850245e+00+2.015583e+00i  -9.9805968042e-02+2.015583e+00i  -1.4732345723e+01+0.000000e+00i
period   2:  +1.1950718171e+00+0.000000e+00i  -2.0042179200e-01+0.000000e+00i  -1.4661316692e+01+0.000000e+00i
period   3:  +1.0304613619e+00+0.000000e+00i  -3.5811336909e-02+0.000000e+00i  -1.4661316691e+01+0.000000e+00i
period   4:  +1.0020068899e+00+0.000000e+00i  -7.3568648900e-03+0.000000e+00i  -1.4661316691e+01+0.000000e+00i
```

The first periods are transients. Before the vectors have converged, real
multipliers may show an imaginary part of π/T, or the vectors of two real
multipliers may rotate into each other. Such a rotation is not taken as a
complex pair, because the eigenvalues of its $2\times 2$ block are real.

## Limitations

- Multipliers of nearly equal modulus converge slowly. Distinct real
  multipliers of equal modulus do not converge.
- A complex pair whose phase is within about `θtol` of 0 or π cannot be told
  apart from two real multipliers.
- The tangent vectors must have a real inner product. `dot` and `norm` are
  used for Gram–Schmidt and for the Procrustes problem, and `dot` must return a
  real number. The state type must also support `similar`, broadcasting and
  `setindex!`, used to build the canonical initial vectors.
- The base orbit must be converged. On an orbit that does not close, the zero
  exponent is only approximately zero.
- The exponents are as accurate as the integration of the linearised
  equations and the interpolation of the orbit.

## Tests

The tests check the exponents of a linear system with constant coefficients
against the eigenvalues of its matrix, for several periods, schemes and
numbers of tangent vectors. They check a linear system with time-periodic
coefficients, $\dot x = R(t) A R(t)^\top x$ with $R(t)$ a rotation, against
its monodromy matrix computed analytically, including negative multipliers.
They check the `shift` keyword on a system that commutes with a rotation.
Finally, they converge the Lorenz orbits AB and AAB by Newton shooting and
check the zero exponent, the sum of the exponents and the leading exponent.
They also converge the orbit with ten loops of the table above, and check its
zero exponent and the sum rule.
Run them with

```julia
julia --project -e 'using Pkg; Pkg.test()'
```

## References

- D. Lasagna, Sensitivity of long periodic orbits of chaotic systems,
  Phys. Rev. E 102, 052220 (2020), [arXiv:1910.06706](https://arxiv.org/abs/1910.06706).
- X. Ding and P. Cvitanović, Periodic eigendecomposition and its application to
  Kuramoto–Sivashinsky system, SIAM J. Appl. Dyn. Syst. 15, 1434 (2016).
- K. Lust, Improved numerical Floquet multipliers, Int. J. Bifurcation Chaos 11,
  2389 (2001).
- D. Viswanath, Symbolic dynamics and periodic orbits of the Lorenz attractor,
  Nonlinearity 16, 1035 (2003).
- G. H. Golub and C. F. Van Loan, Matrix Computations, 4th edition, Johns
  Hopkins University Press (2013), for the orthogonal Procrustes problem.
