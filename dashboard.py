import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# -----------------------------------------------------------------------------
# Physics functions
# -----------------------------------------------------------------------------

def schw_null_geodesics(t, w, M, L):
    r, rdot, phi = w
    phidot = L / r ** 2
    return [rdot, L ** 2 * (r - 3 * M) / r ** 4, phidot]


def expected_schw_light_bending(r, M):
    return 4.0 * M / r


def solve_trajectory(initial, M, L, t_max=10000.0, dt=0.05):
    t_eval = np.arange(0.0, t_max, dt)
    sol = solve_ivp(
        schw_null_geodesics,
        (0.0, t_max),
        initial,
        t_eval=t_eval,
        args=(M, L),
        rtol=1e-9,
        atol=1e-9,
    )
    r, phi = sol.y[0], sol.y[2]
    x, y = r * np.cos(phi), r * np.sin(phi)
    return x, y, t_eval


def calculate_deflection(x, y, t):
    n = len(t)
    grad = (y[n * 4 // 5] - y[-1]) / (x[n * 4 // 5] - x[-1])
    return np.arctan(-grad)


def calculate_initial_conditions(b, initial_x):
    initial_r = np.sqrt(b ** 2 + initial_x ** 2)
    initial_phi = np.arccos(initial_x / initial_r)
    initial_rdot = np.cos(initial_phi)
    initial_phidot = -np.sqrt((1.0 - initial_rdot ** 2) / initial_r ** 2)
    L = initial_r ** 2 * initial_phidot
    return [initial_r, initial_rdot, initial_phi], L

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

    # Event horizon circle
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
