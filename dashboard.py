import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import json
from solved_problem import solve, expected_schw_light_bending, calculate_deflection, calculate_initial_conditions

with open("config.json") as f:
    config = json.load(f)

METHOD = config["method"]

def solve_trajectory(initial, M, L, t_max=1000.0, dt=0.05):
    t_eval = np.arange(0.0, t_max, dt)
    r, phi = solve((0.0, t_max), t_eval, initial, [M, L]) 
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, t_eval[:len(r)] 

# -----------------------------------------------------------------------------
# Styling parameters
# -----------------------------------------------------------------------------

PLOT_TEMPLATE = "plotly_white"
PRIMARY_COLOR = "#0d6efd"     
NUMERICAL_COLOR = "#ff6f00"    

# -----------------------------------------------------------------------------
# Figure builders
# -----------------------------------------------------------------------------

def build_light_ray_figure(M, impact_params, initial_x=-50):
    fig = go.Figure()
    for b in impact_params:
        initial, L = calculate_initial_conditions(b, initial_x)
        x, y, _ = solve_trajectory(initial, M, L)
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", name=f"b = {b}", line=dict(width=2))
        )

    R = 2 * M
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(
        go.Scatter(
            x=R * np.cos(theta),
            y=R * np.sin(theta),
            fill="toself",
            mode="lines",
            line=dict(color="black"),
            fillcolor="black",
            name="Black Hole",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    lim = abs(initial_x)
    fig.update_layout(
        template=PLOT_TEMPLATE,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-lim, lim], title="x (coordinate)"),
        yaxis=dict(range=[-lim/1.5, lim/1.5], title="y (coordinate)"),
        legend_title="Impact parameter b",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )
    return fig


def build_deflection_figure(M, b_min=20, b_max=100, n=40, initial_x=-50):
    bs = np.linspace(b_min, b_max, n)
    numerical = []
    for b in bs:
        initial, L = calculate_initial_conditions(b, initial_x)
        x, y, t = solve_trajectory(initial, M, L)
        numerical.append(calculate_deflection(x, y, t))

    theoretical = expected_schw_light_bending(bs, M)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bs,
            y=theoretical,
            mode="markers+lines",
            marker=dict(symbol="circle", size=8, color=PRIMARY_COLOR),
            line=dict(color=PRIMARY_COLOR, dash="dash"),
            name="Theoretical 4M / b",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bs,
            y=numerical,
            mode="markers",
            marker=dict(symbol="cross", size=9, color=NUMERICAL_COLOR),
            name="Numerical result",
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        xaxis_title="Impact parameter b",
        yaxis_title="Deflection angle (radians)",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )
    return fig

# -----------------------------------------------------------------------------
# Dash application
# -----------------------------------------------------------------------------

external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets, title="Schwarzschild Dashboard")

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H2(
                    "Schwarzschild Light‑Bending Explorer",
                    style={
                        "textAlign": "center",
                        "marginTop": "1.5rem",
                        "marginBottom": "1rem",
                        "fontWeight": "600",
                        "color": "#212529",
                    },
                ),
                width=12,
            )
        ),
        dbc.Row(
            dbc.Col(
                html.P(
                    "Adjust the black‑hole mass M with the slider to see how photon trajectories and deflection angles respond.",
                    style={"textAlign": "center", "margin": "0 auto", "color": "#495057"},
                )
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Slider(
                        id="mass-slider",
                        min=0.5,
                        max=3.5,
                        step=0.05,
                        value=1.0,
                        tooltip={"placement": "bottom", "always_visible": False},
                        marks={i: {"label": str(i), "style": {"color": "#000000"}} for i in range(1, 6)},
                    ),
                    width=12,
                    style={"padding": "1rem 1.5rem"},
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Light Rays Around a Schwarzschild Black Hole"),
                            dbc.CardBody(
                                dcc.Graph(id="light-rays-graph", config={"displayModeBar": False})
                            ),
                        ],
                        class_name="shadow-sm rounded-4",
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Deflection Angle vs Impact Parameter"),
                            dbc.CardBody(
                                dcc.Graph(id="deflection-graph", config={"displayModeBar": False})
                            ),
                        ],
                        class_name="shadow-sm rounded-4",
                    ),
                    md=6,
                ),
            ],
            style={"marginBottom": "2rem"},
        ),
    ],
    fluid=True,
    class_name="px-3",
)


@app.callback(
    Output("light-rays-graph", "figure"),
    Output("deflection-graph", "figure"),
    Input("mass-slider", "value"),
)
def update_figures(mass):
    impact_params = [4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    fig1 = build_light_ray_figure(mass, impact_params)
    fig2 = build_deflection_figure(mass)
    return fig1, fig2


if __name__ == "__main__":
    app.run(debug=True)
