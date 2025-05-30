# Third-Party Library Imports
import plotly.graph_objects as go


def format_plot(fig: go.Figure, xtitle: str, ytitle: str, width: int = 840, height: int = 600):
    """
    Apply consistent formatting to a Plotly figure.

    Args:
        fig (go.Figure): Plotly figure to format.
        xtitle (str): Title of the x-axis.
        ytitle (str): Title of the y-axis.
        width (int, optional): Plot width in pixels. Defaults to 840.
        height (int, optional): Plot height in pixels. Defaults to 600.
    """

    fig.update_layout(
        font=dict(family="DejaVu Sans"),
        plot_bgcolor="white",
        width=width,
        height=height,
        legend=dict(font=dict(size=20, color="black")),
    )
    axis_params = dict(
        showline=True,
        linecolor="black",
        linewidth=1,
        ticks="outside",
        mirror="allticks",
        tickwidth=1,
        tickcolor="black",
        showgrid=False,
        tickfont=dict(color="rgb(0,0,0)", size=20),
    )
    fig.update_xaxes(title=dict(text=xtitle, font=dict(size=22, color="rgb(0,0,0)")), **axis_params)
    fig.update_yaxes(title=dict(text=ytitle, font=dict(size=22, color="rgb(0,0,0)")), **axis_params)
