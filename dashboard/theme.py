"""Dash bootstrap theme and shared style constants."""

import dash_bootstrap_components as dbc

THEME = dbc.themes.DARKLY
ACCENT = "#00d4ff"
BG = "#1a1a2e"
CARD_BG = "#16213e"
TEXT = "#eaeaea"

CARD_STYLE = {
    "background": CARD_BG,
    "border": f"1px solid {ACCENT}",
    "borderRadius": "8px",
    "padding": "16px",
    "marginBottom": "16px",
    "color": TEXT,
}

HEADER_STYLE = {
    "background": BG,
    "borderBottom": f"2px solid {ACCENT}",
    "padding": "12px 24px",
    "marginBottom": "0",
}
