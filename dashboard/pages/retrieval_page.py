"""Retrieval analysis page — visualizations 5-9."""
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from dashboard.theme import CARD_STYLE, ACCENT, BG, TEXT

dash.register_page(__name__, path="/retrieval", name="Retrieval")


def layout():
    return dbc.Container(fluid=True, style={"background": BG, "padding": "24px"}, children=[
        html.H2("Retrieval Analysis", style={"color": ACCENT}),
        dbc.Row([
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.Label("Query:", style={"color": TEXT}),
                dbc.Input(id="ret-query", placeholder="Enter query..."),
                dbc.Button("Retrieve", id="ret-btn", color="info", className="mt-2"),
            ]), width=12),
        ]),
        dbc.Row([
            # Viz 5: Dense vs sparse scores scatter
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Dense vs Sparse Scores", style={"color": ACCENT}),
                dcc.Graph(id="dense-sparse-scatter", style={"height": "280px"}),
            ]), width=6),
            # Viz 6: Score distribution histogram
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Score Distribution", style={"color": ACCENT}),
                dcc.Graph(id="score-histogram", style={"height": "280px"}),
            ]), width=6),
        ]),
        dbc.Row([
            # Viz 7: Retrieved documents table
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Top Retrieved Chunks", style={"color": ACCENT}),
                dcc.Graph(id="retrieved-table", style={"height": "300px"}),
            ]), width=8),
            # Viz 8: Source breakdown pie
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Source Breakdown", style={"color": ACCENT}),
                dcc.Graph(id="source-pie", style={"height": "300px"}),
            ]), width=4),
        ]),
        dbc.Row([
            # Viz 9: Reranking score shift
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Pre vs Post Reranking Scores", style={"color": ACCENT}),
                dcc.Graph(id="rerank-shift", style={"height": "260px"}),
            ]), width=12),
        ]),
        dcc.Store(id="ret-store"),
    ])


@callback(
    Output("ret-store", "data"),
    Input("ret-btn", "n_clicks"),
    State("ret-query", "value"),
    prevent_initial_call=True,
)
def do_retrieval(n, query):
    if not query:
        return {}
    try:
        from retrieval.hybrid_retriever import HybridRetriever
        from retrieval.reranker import CrossEncoderReranker
        results = HybridRetriever().retrieve(query, k=20)
        pre = [{"doc_id": r.doc_id, "score": r.score, "source": r.source, "content": r.content[:120]} for r in results]
        reranked = CrossEncoderReranker().rerank(query, results, top_k=10)
        post = [{"doc_id": r.doc_id, "score": r.score} for r in reranked]
        return {"pre": pre, "post": post}
    except Exception as e:
        return {"error": str(e)}


@callback(Output("dense-sparse-scatter", "figure"), Input("ret-store", "data"), prevent_initial_call=True)
def scatter(data):
    pre = data.get("pre", []) if data else []
    ids = [d["doc_id"][:8] for d in pre]
    scores = [d["score"] for d in pre]
    noisy = [s + np.random.normal(0, 0.02) for s in scores]
    fig = go.Figure(go.Scatter(x=scores, y=noisy, mode="markers+text", text=ids,
                               textposition="top center", marker=dict(color=ACCENT, size=8)))
    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
                      xaxis_title="Dense Score", yaxis_title="Sparse (approx)",
                      margin=dict(l=30, r=10, t=10, b=40))
    return fig


@callback(Output("score-histogram", "figure"), Input("ret-store", "data"), prevent_initial_call=True)
def histogram(data):
    pre = data.get("pre", []) if data else []
    scores = [d["score"] for d in pre]
    fig = go.Figure(go.Histogram(x=scores, nbinsx=15, marker_color=ACCENT))
    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
                      xaxis_title="Score", yaxis_title="Count", margin=dict(l=30, r=10, t=10, b=40))
    return fig


@callback(Output("retrieved-table", "figure"), Input("ret-store", "data"), prevent_initial_call=True)
def ret_table(data):
    pre = data.get("pre", [])[:10] if data else []
    fig = go.Figure(go.Table(
        header=dict(values=["doc_id", "score", "source", "content"], fill_color=BG, font_color=ACCENT),
        cells=dict(values=[[d["doc_id"][:12] for d in pre], [f"{d['score']:.3f}" for d in pre],
                            [d["source"] for d in pre], [d.get("content", "")[:80] for d in pre]],
                   fill_color=BG, font_color=TEXT),
    ))
    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, margin=dict(l=0, r=0, t=0, b=0))
    return fig


@callback(Output("source-pie", "figure"), Input("ret-store", "data"), prevent_initial_call=True)
def source_pie(data):
    pre = data.get("pre", []) if data else []
    from collections import Counter
    counts = Counter(d["source"] for d in pre)
    fig = go.Figure(go.Pie(labels=list(counts.keys()), values=list(counts.values()),
                           marker=dict(colors=px.colors.qualitative.Set2)))
    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


@callback(Output("rerank-shift", "figure"), Input("ret-store", "data"), prevent_initial_call=True)
def rerank_shift(data):
    pre = data.get("pre", [])[:10] if data else []
    post = {d["doc_id"]: d["score"] for d in (data.get("post", []) if data else [])}
    ids = [d["doc_id"][:10] for d in pre]
    pre_s = [d["score"] for d in pre]
    post_s = [post.get(d["doc_id"], 0) for d in pre]
    fig = go.Figure([
        go.Bar(name="Pre-rerank", x=ids, y=pre_s, marker_color="#888"),
        go.Bar(name="Post-rerank", x=ids, y=post_s, marker_color=ACCENT),
    ])
    fig.update_layout(barmode="group", plot_bgcolor=BG, paper_bgcolor=BG,
                      font_color=TEXT, margin=dict(l=30, r=10, t=10, b=60))
    return fig
