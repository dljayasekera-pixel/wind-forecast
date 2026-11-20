"""Streamlit dashboard over backtest artifacts.

    streamlit run app/streamlit_app.py -- --artifacts artifacts/synthetic

Reads only files written by ``windcast backtest``; it never trains or calls a data source.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Fixed colour per entity so a model or provider looks the same on every chart.
MODEL_COLORS = {"stack": "#2a78d6", "lgbm_q50": "#eb6834", "nwp_pc": "#1baf7a", "persistence": "#eda100",
                "xgb": "#e87ba4", "arima": "#008300", "climatology": "#6250d6"}
MODEL_LABELS = {"stack": "Stacked ensemble", "lgbm_q50": "LightGBM median", "nwp_pc": "NWP + power curve",
                "persistence": "Persistence", "xgb": "XGBoost", "arima": "ARIMA", "climatology": "Climatology"}
PROVIDER_COLORS = {"blend": "#2a78d6", "gfs": "#eb6834", "ecmwf": "#1baf7a", "icon": "#eda100"}
EXTRA_COLORS = ["#e87ba4", "#008300", "#6250d6", "#e34948"]
CAUSE_LABELS = {"weather": "Weather (NWP)", "model": "Model", "operations": "Operations",
                "data_quality": "Data quality", "provider_outage": "No NWP available"}
BAND = "#2a78d6"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"


def rounded(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Round float columns only (timestamps are left alone)."""
    floats = df.select_dtypes("float").columns
    return df.assign(**{c: df[c].round(digits) for c in floats})


def artifacts_dir() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--artifacts", default=os.environ.get("WINDCAST_ARTIFACTS", "artifacts/synthetic"))
    return Path(parser.parse_known_args()[0].artifacts)


@st.cache_data(show_spinner=False)
def load(path: str) -> dict:
    root = Path(path) / "backtest"
    out = {"preds": pd.read_parquet(root / "preds.parquet"),
           "summary": json.loads((root / "summary.json").read_text())}
    for csv in (root / "tables").glob("*.csv"):
        out[csv.stem] = pd.read_csv(csv)
    return out


def style(fig: go.Figure, title: str, x: str, y: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"),
        height=height, margin=dict(l=10, r=10, t=48, b=10), hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None))
    fig.update_xaxes(title=x, showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(title=y, gridcolor=GRID, zeroline=False)
    return fig


def fig_fan(preds: pd.DataFrame, issue_time, capacity_mw: float) -> go.Figure:
    """Quantile fan for one forecast issue, with the realised power."""
    g = preds[preds["issue_time"] == issue_time].sort_values("valid_time")
    fig = go.Figure()
    for lo, hi, alpha, name in (("lgbm_q05", "lgbm_q95", 0.18, "5-95% interval"),
                                ("lgbm_q25", "lgbm_q75", 0.32, "25-75% interval")):
        if lo not in g or hi not in g:
            continue
        fig.add_trace(go.Scatter(x=g["valid_time"], y=g[lo] * capacity_mw, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=g["valid_time"], y=g[hi] * capacity_mw, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=f"rgba(42,120,214,{alpha})", name=name,
                                 hovertemplate="%{y:.1f} MW"))
    fig.add_trace(go.Scatter(x=g["valid_time"], y=g["stack"] * capacity_mw, mode="lines+markers",
                             line=dict(color=BAND, width=2), marker=dict(size=5), name="Forecast (P50)",
                             hovertemplate="%{y:.1f} MW"))
    fig.add_trace(go.Scatter(x=g["valid_time"], y=g["y"] * capacity_mw, mode="lines+markers",
                             line=dict(color=INK, width=2, dash="dot"), marker=dict(size=5), name="Actual",
                             hovertemplate="%{y:.1f} MW"))
    title = f"Forecast issued {pd.Timestamp(issue_time):%Y-%m-%d %H:%M} UTC"
    fig = style(fig, title, "Valid time (UTC)", "Farm power (MW)")
    fig.update_yaxes(range=[0, capacity_mw * 1.05])
    return fig


def fig_metric_by_lead(table: pd.DataFrame, metric: str, models: list[str], title: str, ylab: str) -> go.Figure:
    fig = go.Figure()
    for model in models:
        g = table[table["model"] == model].sort_values("lead_h")
        if g.empty:
            continue
        fig.add_trace(go.Scatter(x=g["lead_h"], y=g[metric] * 100, mode="lines+markers",
                                 name=MODEL_LABELS.get(model, model),
                                 line=dict(color=MODEL_COLORS.get(model, MUTED), width=2), marker=dict(size=6),
                                 hovertemplate="%{y:.1f}"))
    fig = style(fig, title, "Lead time (hours)", ylab)
    fig.update_xaxes(type="log" if table["lead_h"].max() > 24 else "linear")
    return fig


def fig_grouped(table: pd.DataFrame, group: str, models: list[str], title: str) -> go.Figure:
    """nMAE by a stratum (season, regime), pooled over lead buckets by row count."""
    t = table[table["model"].isin(models)].copy()
    t["w"] = t["mae"] * t["n"]
    agg = t.groupby([group, "model"], observed=True)[["w", "n"]].sum()
    agg["mae"] = agg["w"] / agg["n"]
    agg = agg.reset_index()
    fig = go.Figure()
    for model in models:
        g = agg[agg["model"] == model]
        fig.add_trace(go.Bar(x=g[group], y=g["mae"] * 100, name=MODEL_LABELS.get(model, model),
                             marker=dict(color=MODEL_COLORS.get(model, MUTED), line=dict(color="#fcfcfb", width=2)),
                             hovertemplate="%{y:.1f}% of capacity"))
    fig = style(fig, title, "", "nMAE (% of capacity)")
    fig.update_layout(barmode="group", bargap=0.3, hovermode="closest")
    return fig


def fig_reliability(rel: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, width=1, dash="dash"),
                             name="Perfect calibration", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rel["nominal"], y=rel["observed"], mode="lines+markers",
                             line=dict(color=BAND, width=2), marker=dict(size=9), name="LightGBM quantiles",
                             hovertemplate="nominal %{x:.2f}<br>observed %{y:.3f}"))
    fig = style(fig, "Quantile reliability", "Nominal quantile level", "Observed frequency below forecast")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def fig_coverage(prob: pd.DataFrame) -> go.Figure:
    nominal = float(prob["nominal_coverage"].iloc[0]) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob["lead_h"], y=prob["coverage"] * 100, mode="lines+markers",
                             line=dict(color=BAND, width=2), marker=dict(size=6), name="Observed coverage",
                             hovertemplate="%{y:.1f}%"))
    fig.add_hline(y=nominal, line=dict(color=MUTED, width=1, dash="dash"),
                  annotation_text=f"nominal {nominal:.0f}%", annotation_font_color=MUTED)
    fig = style(fig, "Coverage of the central interval by lead", "Lead time (hours)", "Coverage (%)")
    fig.update_yaxes(range=[50, 100])
    fig.update_xaxes(type="log" if prob["lead_h"].max() > 24 else "linear")
    return fig


def provider_color(name: str, order: list[str]) -> str:
    if name in PROVIDER_COLORS:
        return PROVIDER_COLORS[name]
    extra = [p for p in order if p not in PROVIDER_COLORS]
    return EXTRA_COLORS[extra.index(name) % len(EXTRA_COLORS)]


def fig_provider(acc: pd.DataFrame, metric: str, title: str, ylab: str, scale: float = 1.0) -> go.Figure:
    order = list(dict.fromkeys(acc["provider"]))
    buckets = list(dict.fromkeys(acc["lead_bucket"]))
    fig = go.Figure()
    for provider in sorted(order, key=lambda p: (p != "blend", p)):
        g = acc[acc["provider"] == provider].set_index("lead_bucket").reindex(buckets)
        fig.add_trace(go.Bar(x=buckets, y=g[metric] * scale, name=provider,
                             marker=dict(color=provider_color(provider, order), line=dict(color="#fcfcfb", width=2)),
                             hovertemplate="%{y:.2f}"))
    fig = style(fig, title, "Lead time", ylab)
    fig.update_layout(barmode="group", bargap=0.3, hovermode="closest")
    return fig


def fig_rca_by_lead(rca: pd.DataFrame) -> go.Figure:
    series = [("weather_mae", "Weather (NWP)", "#2a78d6"), ("model_mae", "Model", "#eb6834"),
              ("operations_mae", "Operations", "#1baf7a")]
    fig = go.Figure()
    for col, label, color in series:
        fig.add_trace(go.Scatter(x=rca["lead_h"], y=rca[col] * 100, mode="lines+markers", name=label,
                                 line=dict(color=color, width=2), marker=dict(size=6), hovertemplate="%{y:.1f}"))
    fig = style(fig, "Mean absolute size of each error component", "Lead time (hours)", "% of capacity")
    fig.update_xaxes(type="log" if rca["lead_h"].max() > 24 else "linear")
    return fig


def fig_causes(summary: pd.DataFrame) -> go.Figure:
    s = summary.sort_values("share_of_big_misses")
    fig = go.Figure(go.Bar(x=s["share_of_big_misses"] * 100, y=s["cause"].map(CAUSE_LABELS).fillna(s["cause"]),
                           orientation="h", marker=dict(color=BAND),
                           text=[f"{v:.0f}%" for v in s["share_of_big_misses"] * 100], textposition="outside",
                           textfont=dict(color=INK), hovertemplate="%{x:.1f}% of big misses"))
    fig = style(fig, "Dominant cause of large forecast misses", "Share of large misses (%)", "", height=300)
    fig.update_layout(hovermode="closest")
    return fig


def main() -> None:
    st.set_page_config(page_title="Wind forecast backtest", layout="wide")
    path = artifacts_dir()
    if not (path / "backtest" / "summary.json").exists():
        st.error(f"No backtest found under '{path}'. Run: windcast run-all --config configs/synthetic.yaml")
        st.stop()
    d = load(str(path))
    s, preds = d["summary"], d["preds"]
    capacity_mw = st.sidebar.number_input("Installed capacity (MW)", value=12.3, min_value=0.1, step=0.1)
    st.sidebar.caption(f"Artifacts: `{path}`")
    st.sidebar.caption(f"Test period {s['test_start'][:10]} to {s['test_end'][:10]}, {s['n_folds']} folds")

    st.title("Wind power forecast: backtest report")
    st.caption("Out-of-sample results from the rolling-origin backtest. Errors are in percent of installed "
               "capacity and exclude hours with outages or curtailment unless stated.")
    c = st.columns(5)
    c[0].metric("Ensemble nMAE", f"{s['stack_nmae'] * 100:.1f}%")
    c[1].metric("Persistence nMAE", f"{s['persistence_nmae'] * 100:.1f}%")
    c[2].metric("RMSE skill vs persistence", f"{s['stack_skill_rmse_vs_persistence'] * 100:.0f}%")
    c[3].metric("CRPS skill vs climatology", f"{s['crpss_vs_climatology'] * 100:.0f}%")
    c[4].metric(f"Coverage of {s['nominal_coverage'] * 100:.0f}% interval", f"{s['interval_coverage'] * 100:.0f}%")

    tabs = st.tabs(["Forecast explorer", "Accuracy", "Uncertainty", "Weather providers", "Root cause"])

    with tabs[0]:
        issues = sorted(preds["issue_time"].unique())
        issue = st.select_slider("Forecast issue time (UTC)", options=issues, value=issues[len(issues) // 2],
                                 format_func=lambda t: f"{pd.Timestamp(t):%Y-%m-%d %H:%M}")
        st.plotly_chart(fig_fan(preds, issue, capacity_mw), width="stretch")
        g = preds[preds["issue_time"] == issue]
        if (g["fallback_tier"] > 0).any():
            st.warning("This forecast ran in fallback mode: at least one weather provider was missing or stale.")
        with st.expander("Data table"):
            cols = ["valid_time", "lead_h", "y", "stack"] + [c for c in g if c.startswith("lgbm_q")] + ["fallback_tier"]
            st.dataframe(rounded(g[cols]), hide_index=True, width="stretch")

    with tabs[1]:
        models = st.multiselect("Models", list(MODEL_LABELS), default=["stack", "lgbm_q50", "nwp_pc", "persistence"],
                                format_func=MODEL_LABELS.get)
        left, right = st.columns(2)
        left.plotly_chart(fig_metric_by_lead(d["metrics_by_lead"], "mae", models, "Error by lead time",
                                             "nMAE (% of capacity)"), width="stretch")
        right.plotly_chart(fig_metric_by_lead(d["metrics_by_lead"], "skill_rmse",
                                              [m for m in models if m != "persistence"],
                                              "Skill over persistence by lead time", "RMSE skill (%)"),
                           width="stretch")
        left, right = st.columns(2)
        left.plotly_chart(fig_grouped(d["metrics_by_season"], "season", models, "Error by season"),
                          width="stretch")
        right.plotly_chart(fig_grouped(d["metrics_by_regime"], "regime", models, "Error by wind regime"),
                           width="stretch")
        with st.expander("Diebold-Mariano tests (ensemble against each alternative)"):
            st.caption("Negative statistic: the ensemble has the lower squared error.")
            st.dataframe(d["dm_tests"], hide_index=True, width="stretch")

    with tabs[2]:
        left, right = st.columns(2)
        left.plotly_chart(fig_reliability(d["reliability"]), width="stretch")
        right.plotly_chart(fig_coverage(d["prob_by_lead"]), width="stretch")
        st.dataframe(d["prob_by_lead"], hide_index=True, width="stretch")

    with tabs[3]:
        acc = d["provider_accuracy"]
        left, right = st.columns(2)
        left.plotly_chart(fig_provider(acc, "ws_rmse_bc", "Wind-speed error after bias correction",
                                       "RMSE (m/s)"), width="stretch")
        right.plotly_chart(fig_provider(acc, "power_nmae", "Power error if each provider were used alone",
                                        "nMAE (% of capacity)", 100), width="stretch")
        st.subheader("Delivery")
        st.dataframe(d["provider_delivery"], hide_index=True, width="stretch")
        left, right = st.columns(2)
        left.subheader("Blend weights")
        left.dataframe(d["blend_weights"], hide_index=True, width="stretch")
        right.subheader("Bias before and after correction (m/s)")
        right.dataframe(acc[acc["provider"] != "blend"][["provider", "lead_bucket", "ws_bias_raw", "ws_bias_bc"]],
                        hide_index=True, width="stretch")

    with tabs[4]:
        left, right = st.columns(2)
        left.plotly_chart(fig_causes(d["rca_summary"]), width="stretch")
        right.plotly_chart(fig_rca_by_lead(d["rca_by_lead"]), width="stretch")
        st.subheader("Largest misses")
        ev = d["rca_events"]
        show = ["issue_time", "valid_time", "lead_h", "y", "forecast", "error", "c_weather", "c_model",
                "c_operations", "tgt_availability", "cause"]
        st.dataframe(rounded(ev[show]).head(50), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
