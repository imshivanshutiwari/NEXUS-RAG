"""Query interface page — visualizations 1-4."""
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input, State
import plotly.graph_objects as go

from dashboard.theme import CARD_STYLE, ACCENT, BG, TEXT

dash.register_page(__name__, path="/query", name="Query")


def layout():
    return dbc.Container(fluid=True, style={"background": BG, "padding": "24px"}, children=[
        html.H2("Query Interface", style={"color": ACCENT}),
        dbc.Row([
            dbc.Col([
                dbc.Card(style=CARD_STYLE, children=[
                    html.Label("Enter your question:", style={"color": TEXT}),
                    dbc.Textarea(id="query-input", placeholder="Ask anything...", style={"height": "100px"}),
                    dbc.Button("Run Query", id="run-query-btn", color="info", className="mt-2"),
                ])
            ], width=12),
        ]),
        dbc.Row([
            # Viz 1: Answer card
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Answer", style={"color": ACCENT}),
                html.Div(id="answer-output", style={"color": TEXT, "whiteSpace": "pre-wrap"}),
            ]), width=8),
            # Viz 2: RAGAS scores gauge
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("RAGAS Scores", style={"color": ACCENT}),
                dcc.Graph(id="ragas-gauge", style={"height": "250px"}),
            ]), width=4),
        ]),
        dbc.Row([
            # Viz 3: Pipeline trace timeline
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Pipeline Trace", style={"color": ACCENT}),
                dcc.Graph(id="pipeline-trace-chart", style={"height": "200px"}),
            ]), width=6),
            # Viz 4: Citations table
            dbc.Col(dbc.Card(style=CARD_STYLE, children=[
                html.H5("Citations", style={"color": ACCENT}),
                html.Div(id="citations-output", style={"color": TEXT}),
            ]), width=6),
        ]),
        dcc.Store(id="query-result-store"),
    ])


@callback(
    Output("query-result-store", "data"),
    Output("answer-output", "children"),
    Input("run-query-btn", "n_clicks"),
    State("query-input", "value"),
    prevent_initial_call=True,
)
def run_query(n_clicks, query):
    if not query:
        return {}, "Please enter a question."
    try:
        from pipeline.main import NEXUSPipeline
        result = NEXUSPipeline().query(query)
        return result, result.get("answer", "No answer generated.")
    except Exception as e:
        return {}, f"Error: {e}"


@callback(
    Output("ragas-gauge", "figure"),
    Input("query-result-store", "data"),
    prevent_initial_call=True,
)
def update_ragas_gauge(data):
    scores = data.get("ragas_scores", {}) if data else {}
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    values = [scores.get(m, 0) for m in metrics]
    fig = go.Figure(go.Bar(
        x=metrics, y=values,
        marker_color=[ACCENT] * 4,
        text=[f"{v:.2f}" for v in values], textposition="auto",
    ))
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        font_color=TEXT, yaxis_range=[0, 1],
        margin=dict(l=10, r=10, t=10, b=40),
    )
    return fig


@callback(
    Output("pipeline-trace-chart", "figure"),
    Input("query-result-store", "data"),
    prevent_initial_call=True,
)
def update_trace(data):
    trace = data.get("pipeline_trace", []) if data else []
    fig = go.Figure(go.Table(
        header=dict(values=["Stage"], fill_color=BG, font_color=ACCENT),
        cells=dict(values=[trace], fill_color=BG, font_color=TEXT),
    ))
    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, margin=dict(l=0, r=0, t=0, b=0))
    return fig


@callback(
    Output("citations-output", "children"),
    Input("query-result-store", "data"),
    prevent_initial_call=True,
)
def update_citations(data):
    citations = data.get("citations", []) if data else []
    if not citations:
        return "No citations."
    items = []
    for c in citations:
        items.append(html.Div([
            html.B(f"[{c.get('citation_number', '?')}] {c.get('source', '')}"),
            html.P(c.get("content", "")[:200] + "...", style={"fontSize": "0.85rem"}),
        ]))
    return items
