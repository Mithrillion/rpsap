import pandas as pd
import holoviews as hv

hv.extension("bokeh")


dfs_ssp = [
    pd.read_csv(f"results/recofit_ssp_v2_omega_0.675_l100_rd{i}.csv")
    for i in ["0", "0.01", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5"]
]

dfs_nasc = [
    pd.read_csv(f"results/recofit_nasc_80_70_190_70_0.04_0.4_900_rd{i}.csv")
    for i in ["0", "0.1", "0.2", "0.3", "0.4", "0.5"]
]

dfs_rpsap = [
    pd.read_csv(
        f"results/recofit_rpsap_6_20_50_1000_20_33_30_900_0.995_0.05_0.04_True_rd{i}.csv"
    )
    for i in ["0", "0.1", "0.2", "0.3", "0.4", "0.5"]
]
labels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
ssp_labels = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

res = (
    pd.DataFrame(
        {labels[i]: df.iloc[:, 1:].agg("mean") for i, df in enumerate(dfs_rpsap)}
    )
    .iloc[4:]
    .T
)

g1 = (
    hv.Curve(
        res, ("index", "random drop rate"), ("masked_f1", "score"), label="f1"
    ).opts(fontscale=2, width=500, height=500)
    * hv.Curve(
        res, ("index", "random drop rate"), ("masked_acc", "score"), label="acc"
    ).opts(line_dash="dashed")
    * hv.Curve(
        res, ("index", "random drop rate"), ("masked_prec", "score"), label="prec"
    ).opts(line_dash="dotted")
    * hv.Curve(
        res, ("index", "random drop rate"), ("masked_rec", "score"), label="rec"
    ).opts(line_dash="dotdash")
).opts(legend_position="bottom_left", title="RPSAP")

res = (
    pd.DataFrame(
        {ssp_labels[i]: df.iloc[:, 1:].agg("mean") for i, df in enumerate(dfs_ssp)}
    )
    .iloc[4:]
    .T
)

g2 = (
    hv.Curve(
        res, ("index", "random drop rate"), ("masked_f1", "score"), label="f1"
    ).opts(fontscale=2, width=500, height=500)
    * hv.Curve(
        res, ("index", "random drop rate"), ("masked_acc", "score"), label="acc"
    ).opts(line_dash="dashed")
    * hv.Curve(
        res, ("index", "random drop rate"), ("masked_prec", "score"), label="prec"
    ).opts(line_dash="dotted")
    * hv.Curve(
        res, ("index", "random drop rate"), ("masked_rec", "score"), label="rec"
    ).opts(line_dash="dotdash")
).opts(legend_position="top_right", title="R-SIMPAD")

res = (
    pd.DataFrame(
        {labels[i]: df.iloc[:, 1:].agg("mean") for i, df in enumerate(dfs_nasc)}
    )
    .iloc[4:]
    .T
)

g3 = (
    hv.Curve(
        res, ("index", "random drop rate"), ("masked_f1", "score"), label="f1"
    ).opts(fontscale=2, width=500, height=500)
    * hv.Curve(
        res, ("index", "random drop rate"), ("masked_acc", "score"), label="acc"
    ).opts(line_dash="dashed")
    * hv.Curve(
        res, ("index", "random drop rate"), ("masked_prec", "score"), label="prec"
    ).opts(line_dash="dotted")
    * hv.Curve(
        res, ("index", "random drop rate"), ("masked_rec", "score"), label="rec"
    ).opts(line_dash="dotdash")
).opts(legend_position="bottom_left", title="NASC")


g1.opts(ylim=(0.3, 1))
g2.opts(ylim=(0.3, 1))
g3.opts(ylim=(0.3, 1))

hv.save(g1, "plots/perf_rpsap.png")
hv.save(g2, "plots/perf_ssp.png")
hv.save(g3, "plots/perf_nasc.png")
