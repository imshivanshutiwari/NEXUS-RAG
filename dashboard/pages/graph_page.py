"""LangGraph pipeline visualizer — visualizations 20-22."""
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input, State
import plotly.graph_objects as go

from dashboard.theme import CARD_STYLE, ACCENT, BG, TEXT

dash.register_page(__name__, path="/graph", name="Graph")

_NODES = ["router", "retriever", "reranker", "generator", "evaluator", "healer"]
_EDGES = [("router","retriever"),("retriever","reranker"),("reranker","generator"),
          ("generator","evaluator"),("evaluator","healer"),("healer","retriever")]
_NODE_X = [0.1, 0.3, 0.5, 0.7, 0.85, 0.6]
_NODE_Y = [0.5, 0.5, 0.5, 0.5, 0.5, 0.2]


def layout():
    return dbc.Container(fluid=True, style={"background": BG, "padding": "24px"}, children=[
        html.H2("LangGraph Pipeline Visualizer", style={"color": ACCENT}),
        dbc.Row([
            # Viz 20: Static graph topology
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("6-Node Pipeline Graph", style={"color": ACCENT}),
                dcc.Graph(id="graph-topology", figure=_make_topology(), style={"height": "340px"}),
            ]), width=12),
        ]),
        dbc.Row([
            # Viz 21: Active query live trace
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Live Stage Progress", style={"color": ACCENT}),
                dbc.Input(id="graph-query", placeholder="Enter query to trace...", className="mb-2"),
                dbc.Button("Trace", id="trace-btn", color="info"),
                dcc.Graph(id="stage-progress", style={"height": "200px"}),
            ]), width=6),
            # Viz 22: Healing iteration flow
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Healing Iteration Details", style={"color": ACCENT}),
                dcc.Graph(id="healing-flow", style={"height": "200px"}),
                html.Div(id="trace-text", style={"color": TEXT, "fontSize": "0.8rem", "marginTop": "8px"}),
            ]), width=6),
        ]),
    ])


def _make_topology():
    edge_x, edge_y = [], []
    for src, dst in _EDGES:
        xi, yi = _NODE_X[_NODES.index(src)], _NODE_Y[_NODES.index(src)]
        xj, yj = _NODE_X[_NODES.index(dst)], _NODE_Y[_NODES.index(dst)]
        edge_x += [xi, xj, None]; edge_y += [yi, yj, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color="#555", width=2), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=_NODE_X, y=_NODE_Y, mode="markers+text",
                             text=_NODES, textposition="top center",
                             marker=dict(size=24, color=ACCENT, line=dict(color=BG, width=2)),
                             textfont=dict(color=TEXT, size=12)))
    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


@callback(
    Output("stage-progress", "figure"), Output("healing-flow", "figure"), Output("trace-text", "children"),
    Input("trace-btn", "n_clicks"), State("graph-query", "value"), prevent_initial_call=True,
)
def trace_query(n, query):
    if not query:
        return go.Figure(), go.Figure(), ""
    try:
        from pipeline.main import NEXUSPipeline
        result = NEXUSPipeline().query(query)
        trace = result.get("pipeline_trace", [])
        stages_hit = [t.split(":")[0] for t in trace]
        stage_order = [s for s in _NODES if s in stages_hit]
        colors = [ACCENT if s in stages_hit else "#333" for s in _NODES]
        fig1 = go.Figure(go.Bar(x=_NODES, y=[1 if s in stages_hit else 0.2 for s in _NODES],
                                marker_color=colors, showlegend=False))
        fig1.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
                           yaxis_range=[0, 1.2], margin=dict(l=10, r=10, t=10, b=40))
        heals = result.get("healing_attempts", 0)
        fig2 = go.Figure(go.Indicator(mode="number+delta",
                                      value=heals, delta={"reference": 0},
                                      number={"font": {"color": ACCENT, "size": 48}},
                                      title={"text": "Healing Attempts", "font": {"color": TEXT}}))
        fig2.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, margin=dict(l=10, r=10, t=10, b=10))
        trace_str = "\n".join(trace)
        return fig1, fig2, trace_str
    except Exception as e:
        return go.Figure(), go.Figure(), f"Error: {e}"
