from sigtools.sigconv import *
from sigtools.transforms import *
from sigtools.signed_areas import *

import holoviews as hv

hv.extension("bokeh")


def warp(x, factor):
    xp = x / factor
    xpf = np.floor(xp)
    return factor * np.where(
        xpf % 2 == 0, (xp - xpf) ** 1.5 + xpf, (xp - xpf) ** (1 / 1.5) + xpf
    )


xs = np.arange(600)
dfSyn = np.stack(
    [
        0.75 * np.cos(1 / 2 * np.pi / 29 * warp(xs, 67))
        + 0.66 * np.sin(3 / 2 * np.pi / 29 * warp(xs, 67)),
        0.75 * np.sin(1 / 2 * np.pi / 29 * warp(xs, 67) + np.pi / 4)
        + 0.66 * np.cos(3 / 2 * np.pi / 29 * warp(xs, 67) + np.pi / 4),
    ],
    -1,
)

SA = signatory.signature(torch.tensor(dfSyn[None, ...]), 2, True).squeeze()

key_points = [100, 215, 400, 460]
key_points_2 = [50, 500]
states = (
    hv.VLine(key_points[0]).opts(color="green", line_dash="dashed")
    * hv.VLine(key_points[1]).opts(color="green", line_dash="dashed")
    * hv.VLine(key_points[2]).opts(color="green", line_dash="dashed")
    * hv.VLine(key_points[3]).opts(color="green", line_dash="dashed")
)
states_2 = hv.VLine(key_points_2[0]).opts(
    color="purple", line_dash="dashed"
) * hv.VLine(key_points_2[1]).opts(color="purple", line_dash="dashed")
An = hv.Text(float(key_points[0]) + 10, -1, "A", fontsize=15)
Bn = hv.Text(float(key_points[1]) + 10, -1, "B", fontsize=15)
Cn = hv.Text(float(key_points[2]) + 10, -1, "C", fontsize=15)
Dn = hv.Text(float(key_points[3]) + 10, -1, "D", fontsize=15)
Pn = hv.Text(float(key_points_2[0]) + 10, 1, "P", fontsize=15)
Qn = hv.Text(float(key_points_2[1]) + 10, 1, "Q", fontsize=15)
g1 = (
    hv.Curve(dfSyn[:, 0]).opts(width=400, height=220, fontscale=1.5)
    * hv.Curve(dfSyn[:, 1])
    * states
    * states_2
    * An
    * Bn
    * Cn
    * Dn
    * Pn
    * Qn
)
g1

from bokeh.io import export_svgs

p = hv.render(g1, backend="bokeh")
p.output_backend = "svg"
export_svgs(p, filename="plots/curves_example.svg")

seg1 = hv.Segments(
    [
        (
            SA[key_points[0], 3],
            SA[key_points[0], 4],
            SA[key_points[1], 3],
            SA[key_points[1], 4],
        )
    ],
    label="AB",
).opts(color="red", line_width=3, alpha=0.8)
seg2 = hv.Segments(
    [
        (
            SA[key_points[0], 3],
            SA[key_points[0], 4],
            SA[key_points[2], 3],
            SA[key_points[2], 4],
        )
    ],
    label="AC",
).opts(color="green", line_dash="dashed", line_width=3, alpha=0.8)
seg3 = hv.Segments(
    [
        (
            SA[key_points[0], 3],
            SA[key_points[0], 4],
            SA[key_points[3], 3],
            SA[key_points[3], 4],
        )
    ],
    label="AD",
).opts(color="orange", line_dash="dotted", line_width=3, alpha=0.8)
seg_par = hv.Segments(
    [
        (
            SA[key_points_2[0], 3],
            SA[key_points_2[0], 4],
            SA[key_points_2[1], 3],
            SA[key_points_2[1], 4],
        )
    ],
    label="PQ",
).opts(color="purple", line_dash="dotdash", line_width=3, alpha=0.8)
start = hv.Text(0, -0.5, "t=0", fontsize=15)
end = hv.Text(-7.5, 8.5, "t=600", fontsize=15)
A = hv.Text(float(SA[key_points[0], 3]), float(SA[key_points[0], 4]), "A", fontsize=15)
B = hv.Text(float(SA[key_points[1], 3]), float(SA[key_points[1], 4]), "B", fontsize=15)
C = hv.Text(float(SA[key_points[2], 3]), float(SA[key_points[2], 4]), "C", fontsize=15)
D = hv.Text(float(SA[key_points[3], 3]), float(SA[key_points[3], 4]), "D", fontsize=15)
P = hv.Text(
    float(SA[key_points_2[0], 3]), float(SA[key_points_2[0], 4]), "P", fontsize=15
)
Q = hv.Text(
    float(SA[key_points_2[1], 3]), float(SA[key_points_2[1], 4]), "Q", fontsize=15
)
g2 = (
    hv.Curve(SA[:, [3, 4]], "SA1", "SA2").opts(width=400, height=220, fontscale=1.5)
    * seg1
    * seg2
    * seg3
    * start
    * end
    * A
    * B
    * C
    * D
    * P
    * Q
    * seg_par
)
g2 = g2.opts(show_legend=False)
g2

p = hv.render(g2, backend="bokeh")
p.output_backend = "svg"
export_svgs(p, filename="plots/signed_areas_example.svg")
