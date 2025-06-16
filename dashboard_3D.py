import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# -----------------------------------------------------------------------------
# Physics functions
# -----------------------------------------------------------------------------

def schw_null_geodesics(t, w, M, L):
    """Equations of motion for a null geodesic in the equatorial plane (θ = π/2)."""
    r, rdot, phi = w
    phidot = L / r ** 2
    rddot = L ** 2 * (r - 3 * M) / r ** 4
    return [rdot, rddot, phidot]


def expected_schw_light_bending(r, M):
    """First‑order weak‑field deflection: 4M / b."""
    return 4.0 * M / r


def solve_trajectory(initial, M, L, t_max=10000.0, dt=0.05):
    """Integrate a null geodesic and return Cartesian (x, y) plus affine λ."""
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
    """Compute asymptotic deflection angle at large λ."""
    n = len(t)
    grad = (y[n * 4 // 5] - y[-1]) / (x[n * 4 // 5] - x[-1])
    return np.arctan(-grad)


def calculate_initial_conditions(b, initial_x):
    """Return (state, L) for a given impact parameter b and start x‑position."""
    initial_r = np.sqrt(b ** 2 + initial_x ** 2)
    initial_phi = np.arccos(initial_x / initial_r)
    initial_rdot = np.cos(initial_phi)
    initial_phidot = -np.sqrt((1.0 - initial_rdot ** 2) / initial_r ** 2)
    L = initial_r ** 2 * initial_phidot
    return [initial_r, initial_rdot, initial_phi], L


def rotate_xy_plane_to_inclination(x: np.ndarray, y: np.ndarray, i_deg: float):
    """Rotate (x, y, 0) about the x‑axis by inclination i (degrees)."""
    i_rad = np.deg2rad(i_deg)
    y_rot = y * np.cos(i_rad)
    z_rot = y * np.sin(i_rad)
    return x, y_rot, z_rot


# -----------------------------------------------------------------------------
# Styling parameters
# -----------------------------------------------------------------------------

PLOT_TEMPLATE = "plotly_white"
PRIMARY_COLOR = "#0d6efd"     
NUMERICAL_COLOR = "#ff6f00"     

# -----------------------------------------------------------------------------
# Figure builders
# -----------------------------------------------------------------------------

def build_light_ray_figure(M, impact_params, i_deg, initial_x=-50):
    fig = go.Figure()

    for b in impact_params:
        initial, L = calculate_initial_conditions(b, initial_x)
        x, y, _ = solve_trajectory(initial, M, L)
        x3, y3, z3 = rotate_xy_plane_to_inclination(x, y, i_deg)
        fig.add_trace(
            go.Scatter3d(
                x=x3,
                y=y3,
                z=z3,
                mode="lines",
                name=f"b = {b}",
                line=dict(width=3),
            )
        )

    # Event horizon (r = 2M) sphere
    R = 2 * M
    theta = np.linspace(0, 2 * np.pi, 50)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    xh = R * np.sin(phi) * np.cos(theta)
    yh = R * np.sin(phi) * np.sin(theta)
    zh = R * np.cos(phi)

    fig.add_trace(
        go.Surface(
            x=xh,
            y=yh,
            z=zh,
            colorscale=[[0, "black"], [1, "black"]],
            showscale=False,
            opacity=1.0,
            name="Event Horizon",
            hoverinfo="skip",
        )
    )

    lim = abs(initial_x)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-lim, lim], title="x"),
            yaxis=dict(range=[-lim, lim], title="y"),
            zaxis=dict(range=[-lim, lim], title="z"),  
            aspectmode="cube",  #
        ),
        template=PLOT_TEMPLATE,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title="Impact parameter b",
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
        font=dict(family="Inter, sans‑serif"),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )
    return fig


# -----------------------------------------------------------------------------
# Dash application
# -----------------------------------------------------------------------------

external_stylesheets = [dbc.themes.FLATLY]
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    title="Schwarzschild 3‑D Dashboard",
)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H2(
                    "Schwarzschild Light‑Bending Explorer — 3‑D",  # en‑dash
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
                    [
                        "Adjust the mass ", html.Code("M"), ", plus the ",
                        html.Code("inclination"),
                        " of the orbital plane, and watch how photon trajectories and",
                        " deflection angles respond.",
                    ],
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
                    md=6,
                    style={"padding": "1rem 1.5rem"},
                ),
                dbc.Col(
                    dcc.Slider(
                        id="inclination-slider",
                        min=0,
                        max=90,
                        step=1,
                        value=0,
                        tooltip={"placement": "bottom", "always_visible": False},
                        marks={0: {"label": "0°"}, 30: {"label": "30°"}, 60: {"label": "60°"}, 90: {"label": "90°"}},
                    ),
                    md=6,
                    style={"padding": "1rem 1.5rem"},
                ),
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


# -----------------------------------------------------------------------------
# Callback
# -----------------------------------------------------------------------------

@app.callback(
    Output("light-rays-graph", "figure"),
    Output("deflection-graph", "figure"),
    Input("mass-slider", "value"),
    Input("inclination-slider", "value"),
)

def update_figures(mass: float, inclination: float):
    impact_params = [4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    fig1 = build_light_ray_figure(mass, impact_params, inclination)
    fig2 = build_deflection_figure(mass)
    return fig1, fig2


if __name__ == "__main__":
    app.run(debug=True)
