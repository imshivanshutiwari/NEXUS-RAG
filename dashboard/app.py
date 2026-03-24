"""NEXUS-RAG Dash dashboard entry point."""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

from dashboard.theme import THEME, BG, ACCENT, HEADER_STYLE

app = dash.Dash(
    __name__,
    external_stylesheets=[THEME],
    suppress_callback_exceptions=True,
    title="NEXUS-RAG Dashboard",
    use_pages=True,
)
server = app.server

app.layout = dbc.Container(
    fluid=True,
    style={"background": BG, "minHeight": "100vh", "padding": "0"},
    children=[
        dcc.Location(id="url"),
        dbc.NavbarSimple(
            brand="⚡ NEXUS-RAG",
            brand_style={"color": ACCENT, "fontWeight": "bold", "fontSize": "1.4rem"},
            color="dark",
            dark=True,
            style=HEADER_STYLE,
            children=[
                dbc.NavItem(dbc.NavLink("Query", href="/query")),
                dbc.NavItem(dbc.NavLink("Retrieval", href="/retrieval")),
                dbc.NavItem(dbc.NavLink("Evaluation", href="/evaluation")),
                dbc.NavItem(dbc.NavLink("Monitoring", href="/monitoring")),
                dbc.NavItem(dbc.NavLink("Graph", href="/graph")),
            ],
        ),
        dash.page_container,
    ],
)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
