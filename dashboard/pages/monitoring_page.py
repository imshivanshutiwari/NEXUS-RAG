"""Monitoring & drift page — visualizations 15-19."""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input
import plotly.graph_objects as go

from dashboard.theme import CARD_STYLE, ACCENT, BG, TEXT

dash.register_page(__name__, path="/monitoring", name="Monitoring")


def layout():
    return dbc.Container(
        fluid=True,
        style={"background": BG, "padding": "24px"},
        children=[
            html.H2("Monitoring & Drift Detection", style={"color": ACCENT}),
            dcc.Interval(id="mon-interval", interval=5000, n_intervals=0),
            dbc.Row(
                [
                    # Viz 15: RAGAS rolling means line chart
                    dbc.Col(
                        dbc.Card(
                            style=CARD_STYLE,
                            children=[
                                html.H5("RAGAS Rolling Means", style={"color": ACCENT}),
                                dcc.Graph(
                                    id="rolling-means-chart", style={"height": "260px"}
                                ),
                            ],
                        ),
                        width=8,
                    ),
                    # Viz 16: Alert severity gauge
                    dbc.Col(
                        dbc.Card(
                            style=CARD_STYLE,
                            children=[
                                html.H5("Drift Score", style={"color": ACCENT}),
                                dcc.Graph(id="drift-gauge", style={"height": "260px"}),
                            ],
                        ),
                        width=4,
                    ),
                ]
            ),
            dbc.Row(
                [
                    # Viz 17: Latency waterfall per stage
                    dbc.Col(
                        dbc.Card(
                            style=CARD_STYLE,
                            children=[
                                html.H5(
                                    "Stage Latency (P50 / P95)", style={"color": ACCENT}
                                ),
                                dcc.Graph(id="latency-bar", style={"height": "260px"}),
                            ],
                        ),
                        width=6,
                    ),
                    # Viz 18: Healing attempts over time
                    dbc.Col(
                        dbc.Card(
                            style=CARD_STYLE,
                            children=[
                                html.H5("Recent Alerts", style={"color": ACCENT}),
                                html.Div(
                                    id="alert-list",
                                    style={"color": TEXT, "fontSize": "0.85rem"},
                                ),
                            ],
                        ),
                        width=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    # Viz 19: Quality trend line
                    dbc.Col(
                        dbc.Card(
                            style=CARD_STYLE,
                            children=[
                                html.H5("Faithfulness Trend", style={"color": ACCENT}),
                                dcc.Graph(
                                    id="faithfulness-trend", style={"height": "220px"}
                                ),
                            ],
                        ),
                        width=12,
                    ),
                ]
            ),
        ],
    )


def _get_monitor_data():
    try:
        import requests

        r = requests.get("http://localhost:8000/monitor/", timeout=2)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {
        "drift_score": 0.0,
        "drift_status": "STABLE",
        "rolling_faithfulness": 0.82,
        "rolling_relevancy": 0.78,
        "latency_stats": {
            "router": {"p50_ms": 12, "p95_ms": 35},
            "retriever": {"p50_ms": 220, "p95_ms": 480},
            "reranker": {"p50_ms": 110, "p95_ms": 290},
            "generator": {"p50_ms": 850, "p95_ms": 1800},
            "evaluator": {"p50_ms": 60, "p95_ms": 140},
        },
        "recent_alerts": [],
    }


@callback(Output("rolling-means-chart", "figure"), Input("mon-interval", "n_intervals"))
def rolling_means(n):
    from monitoring.quality_monitor import QualityMonitor

    history = QualityMonitor().get_history() or [
        {
            "faithfulness": 0.8,
            "answer_relevancy": 0.75,
            "context_precision": 0.7,
            "context_recall": 0.65,
        }
    ]
    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    colors = [ACCENT, "#ff6b6b", "#ffd93d", "#6bcb77"]
    fig = go.Figure()
    for m, c in zip(metrics, colors):
        vals = [h.get(m, 0) for h in history]
        fig.add_trace(go.Scatter(y=vals, name=m, line=dict(color=c)))
    fig.update_layout(
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color=TEXT,
        yaxis_range=[0, 1],
        margin=dict(l=30, r=10, t=10, b=40),
    )
    return fig


@callback(Output("drift-gauge", "figure"), Input("mon-interval", "n_intervals"))
def drift_gauge(n):
    data = _get_monitor_data()
    score = data.get("drift_score", 0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#ff4d4d" if score > 0.3 else ACCENT},
                "bgcolor": BG,
                "bordercolor": ACCENT,
            },
            number={"font": {"color": TEXT}},
        )
    )
    fig.update_layout(
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color=TEXT,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


@callback(Output("latency-bar", "figure"), Input("mon-interval", "n_intervals"))
def latency_bar(n):
    data = _get_monitor_data()
    stats = data.get("latency_stats", {})
    stages = list(stats.keys())
    p50 = [stats[s].get("p50_ms", 0) for s in stages]
    p95 = [stats[s].get("p95_ms", 0) for s in stages]
    fig = go.Figure(
        [
            go.Bar(name="P50", x=stages, y=p50, marker_color="#6bcb77"),
            go.Bar(name="P95", x=stages, y=p95, marker_color="#ff6b6b"),
        ]
    )
    fig.update_layout(
        barmode="group",
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color=TEXT,
        yaxis_title="ms",
        margin=dict(l=30, r=10, t=10, b=40),
    )
    return fig


@callback(Output("alert-list", "children"), Input("mon-interval", "n_intervals"))
def alert_list(n):
    from monitoring.alert_manager import AlertManager

    alerts = AlertManager().get_recent_alerts(10)
    if not alerts:
        return html.P("No recent alerts.", style={"color": "#6bcb77"})
    return [
        html.Div(
            f"[{a.severity}] {a.name}: {a.message}",
            style={
                "borderLeft": f"3px solid {'#ff4d4d' if a.severity == 'CRITICAL' else '#ffd93d'}",
                "paddingLeft": "8px",
                "marginBottom": "6px",
            },
        )
        for a in alerts
    ]


@callback(Output("faithfulness-trend", "figure"), Input("mon-interval", "n_intervals"))
def faithfulness_trend(n):
    from monitoring.quality_monitor import QualityMonitor

    history = QualityMonitor().get_history()
    vals = [h.get("faithfulness", 0) for h in history] or [0.82]
    threshold = 0.7
    fig = go.Figure(
        [
            go.Scatter(
                y=vals,
                mode="lines+markers",
                line=dict(color=ACCENT),
                name="faithfulness",
            ),
            go.Scatter(
                y=[threshold] * len(vals),
                mode="lines",
                line=dict(color="#ff4d4d", dash="dash"),
                name="threshold",
            ),
        ]
    )
    fig.update_layout(
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color=TEXT,
        yaxis_range=[0, 1],
        margin=dict(l=30, r=10, t=10, b=40),
    )
    return fig
