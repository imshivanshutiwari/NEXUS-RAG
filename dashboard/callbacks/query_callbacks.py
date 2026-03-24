"""Shared query callbacks registered globally."""

from dash import callback, Output, Input, State
from dashboard.theme import ACCENT, BG, TEXT


def register(app):
    """Register extra app-level callbacks not tied to a single page."""
    pass  # Page-level callbacks are registered in each page module.
