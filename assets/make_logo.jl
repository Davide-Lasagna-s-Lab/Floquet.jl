# Draws assets/logo.svg: a periodic orbit with two lobes carrying a ball of perturbations, which
# the flow stretches into ellipses along the unstable direction and squeezes along the stable one.
#
#     julia assets/make_logo.jl

using Printf


# ---- the orbit, a figure of eight ----

A = 124.0   # half width of the loop
B = 62.0    # half height of each lobe

# point and tangent angle, in degrees, of the loop at the phase s
function loop(s)
    x  = A * sin(s)
    y  = B * sin(2s)
    dx = A * cos(s)
    dy = 2B * cos(2s)
    return x, y, atan(dy, dx) * 180 / π
end

points = [loop(2π * k / 600) for k in 0:599]
path   = "M" * join([@sprintf("%.1f %.1f", p[1], p[2]) for p in points], " L") * " Z"


# ---- the perturbations, from a circle to a thin ellipse ----

COLORS  = ["#389826", "#4063d8", "#9558b2", "#cb3c33"]
phases  = [1.50π, 1.86π, 0.30π, 0.68π]   # one per ellipse, in the direction of the flow
radius  = 15.0                           # radius of the initial ball
stretch = [1.0, 1.7, 2.5, 3.3]           # growth along the unstable direction
tilt    = [0.0, 32.0, 40.0, 46.0]        # angle of the unstable direction to the orbit

ellipses = String[]
for i in eachindex(phases)
    x, y, θ = loop(phases[i])
    a = radius * stretch[i]
    b = radius / stretch[i]^0.75
    push!(ellipses,
          @sprintf("<g transform=\"translate(%.1f %.1f) rotate(%.1f)\">", x, y, θ + tilt[i]) *
          @sprintf("<ellipse rx=\"%.1f\" ry=\"%.1f\" fill=\"%s\" fill-opacity=\"0.20\" stroke=\"%s\" stroke-width=\"4.5\"/>",
                   a, b, COLORS[i], COLORS[i]) *
          "<circle r=\"4.5\" fill=\"#242936\"/></g>")
end

# a short arc of the loop, ahead of the first ellipse, that shows the direction of the flow
arrow = "M" * join([@sprintf("%.1f %.1f", loop(s)[1:2]...) for s in range(1.60π, 1.70π; length=8)], " L")


# ---- the svg ----

svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 300" role="img" aria-labelledby="title desc">
  <title id="title">Lyapunov.jl logo</title>
  <desc id="desc">A periodic orbit shaped as a figure of eight carries a ball of perturbations, which the flow stretches into longer and thinner ellipses in Julia colors, beside the Lyapunov.jl wordmark.</desc>
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#f2f5f8"/>
    </linearGradient>
    <filter id="panel-shadow" x="-12%" y="-25%" width="124%" height="150%">
      <feDropShadow dx="0" dy="9" stdDeviation="11" flood-color="#172033" flood-opacity="0.14"/>
    </filter>
    <filter id="soft" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#172033" flood-opacity="0.10"/>
    </filter>
    <marker id="flow-arrow" viewBox="0 0 12 10" refX="6" refY="5" markerWidth="4.4" markerHeight="3.7" orient="auto">
      <path d="M0 0 L12 5 L0 10 Z" fill="#242936"/>
    </marker>
  </defs>

  <!-- Main panel -->
  <rect x="24" y="24" width="912" height="252" rx="24" fill="url(#bg)" filter="url(#panel-shadow)"/>

  <!-- Left illustration: perturbations stretched along a periodic orbit -->
  <g transform="translate(206 150)" filter="url(#soft)">
    <path d="$path" fill="none" stroke="#98a2b3" stroke-width="5" stroke-linejoin="round"/>
    <path d="$arrow" fill="none" stroke="#242936" stroke-width="5" stroke-linecap="round" marker-end="url(#flow-arrow)"/>
    $(join(ellipses, "\n    "))
  </g>

  <!-- Wordmark and subtitle -->
  <g transform="translate(372 96)">
    <text x="0" y="48" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="58" font-weight="820" fill="#222734" letter-spacing="0">Lyapunov.jl</text>
    <text x="2" y="90" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">Lyapunov and Floquet exponents of chaotic flows</text>
    <text x="2" y="118" font-family="Inter, Avenir, Helvetica, Arial, sans-serif"
          font-size="21" font-weight="700" fill="#4f596b" letter-spacing="0">and of their long periodic orbits.</text>
  </g>
</svg>
"""

write(joinpath(@__DIR__, "logo.svg"), svg)
println("written assets/logo.svg")
