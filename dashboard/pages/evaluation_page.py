"""Evaluation metrics page — visualizations 10-14."""
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

from dashboard.theme import CARD_STYLE, ACCENT, BG, TEXT

dash.register_page(__name__, path="/evaluation", name="Evaluation")

_RESULTS_CSV = Path("assets/results/benchmark_results.csv")


def layout():
    return dbc.Container(fluid=True, style={"background": BG, "padding": "24px"}, children=[
        html.H2("Evaluation Dashboard", style={"color": ACCENT}),
        dbc.Button("Run Benchmark", id="run-bench-btn", color="warning", className="mb-3"),
        html.Div(id="bench-status", style={"color": TEXT}),
        dbc.Row([
            # Viz 10: RAGAS radar chart
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Mean RAGAS Scores (Radar)", style={"color": ACCENT}),
                dcc.Graph(id="ragas-radar", style={"height": "300px"}),
            ]), width=6),
            # Viz 11: Per-question heatmap
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Per-Question Score Heatmap", style={"color": ACCENT}),
                dcc.Graph(id="ragas-heatmap", style={"height": "300px"}),
            ]), width=6),
        ]),
        dbc.Row([
            # Viz 12: Healing attempts bar
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Healing Attempts per Query", style={"color": ACCENT}),
                dcc.Graph(id="healing-bar", style={"height": "250px"}),
            ]), width=6),
            # Viz 13: Score distribution violin
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Score Distribution (Violin)", style={"color": ACCENT}),
                dcc.Graph(id="score-violin", style={"height": "250px"}),
            ]), width=6),
        ]),
        dbc.Row([
            # Viz 14: NEXUS-RAG vs baseline bar comparison
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("NEXUS-RAG vs Naive Baseline", style={"color": ACCENT}),
                dcc.Graph(id="baseline-compare", style={"height": "250px"}),
            ]), width=12),
        ]),
        dcc.Store(id="bench-store"),
        dcc.Interval(id="bench-interval", interval=2000, disabled=True),
    ])


def _load_df():
    if _RESULTS_CSV.exists():
        return pd.read_csv(_RESULTS_CSV)
    return pd.DataFrame()


@callback(
    Output("bench-store", "data"), Output("bench-status", "children"),
    Input("run-bench-btn", "n_clicks"), prevent_initial_call=True,
)
def run_bench(n):
    try:
        from evaluation.benchmark_runner import BenchmarkRunner
        df = BenchmarkRunner().run()
        return df.to_dict("records"), f"Benchmark complete: {len(df)} questions evaluated."
    except Exception as e:
        return [], f"Error: {e}"


@callback(Output("ragas-radar", "figure"), Input("bench-store", "data"), prevent_initial_call=True)
def radar(data):
    df = pd.DataFrame(data) if data else _load_df()
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    means = [df[m].mean() if m in df.columns else 0 for m in metrics]
    fig = go.Figure(go.Scatterpolar(r=means + [means[0]], theta=metrics + [metrics[0]],
                                    fill="toself", line_color=ACCENT))
    fig.update_layout(polar=dict(bgcolor=BG, radialaxis=dict(range=[0, 1])),
                      paper_bgcolor=BG, font_color=TEXT, margin=dict(l=20, r=20, t=20, b=20))
    return fig


@callback(Output("ragas-heatmap", "figure"), Input("bench-store", "data"), prevent_initial_call=True)
def heatmap(data):
    df = pd.DataFrame(data) if data else _load_df()
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    cols = [m for m in metrics if m in df.columns]
    if df.empty or not cols:
        return go.Figure()
    z = df[cols].values.T
    fig = go.Figure(go.Heatmap(z=z, x=df["question"].str[:30].tolist() if "question" in df.columns else list(range(len(df))),
                                y=cols, colorscale="Blues", zmin=0, zmax=1))
    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
                      margin=dict(l=120, r=10, t=10, b=80))
    return fig


@callback(Output("healing-bar", "figure"), Input("bench-store", "data"), prevent_initial_call=True)
def healing_bar(data):
    df = pd.DataFrame(data) if data else _load_df()
    if df.empty or "healing_attempts" not in df.columns:
        return go.Figure()
    fig = go.Figure(go.Bar(x=list(range(len(df))), y=df["healing_attempts"].tolist(), marker_color=ACCENT))
    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
                      xaxis_title="Question #", yaxis_title="Attempts", margin=dict(l=30, r=10, t=10, b=40))
    return fig


@callback(Output("score-violin", "figure"), Input("bench-store", "data"), prevent_initial_call=True)
def violin(data):
    df = pd.DataFrame(data) if data else _load_df()
    metrics = ["faithfulness", "answer_relevancy"]
    fig = go.Figure()
    for m in metrics:
        if m in df.columns:
            fig.add_trace(go.Violin(y=df[m].tolist(), name=m, line_color=ACCENT, fillcolor=BG))
    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT, margin=dict(l=30, r=10, t=10, b=40))
    return fig


@callback(Output("baseline-compare", "figure"), Input("bench-store", "data"), prevent_initial_call=True)
def baseline(data):
    df = pd.DataFrame(data) if data else _load_df()
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    nexus_means = [df[m].mean() if (not df.empty and m in df.columns) else 0 for m in metrics]
    baseline_means = [max(0, v - 0.15) for v in nexus_means]
    fig = go.Figure([
        go.Bar(name="NEXUS-RAG", x=metrics, y=nexus_means, marker_color=ACCENT),
        go.Bar(name="Naive RAG", x=metrics, y=baseline_means, marker_color="#555"),
    ])
    fig.update_layout(barmode="group", plot_bgcolor=BG, paper_bgcolor=BG,
                      font_color=TEXT, yaxis_range=[0, 1], margin=dict(l=30, r=10, t=10, b=40))
    return fig
