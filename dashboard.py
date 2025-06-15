# app.py
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from scipy.integrate import solve_ivp


def schw_null_geodesics(t, w, M, L):
    r, rdot, phi = w
    phidot = L / r**2
    return [
        rdot,
        L**2 * (r - 3*M) / r**4,
        phidot
    ]

def expected_schw_light_bending(r, M):
    return 4. * M / r

def solve(t_span, t_eval, initial, p):
    M, L = p
    sol = solve_ivp(schw_null_geodesics, t_span, initial, t_eval=t_eval, args=(M, L), rtol=1e-9, atol=1e-9)
    r = sol.y[0]
    phi = sol.y[2]
    return r, phi

def calculate_deflection(x, y, t):
    num_entries = len(t)
    gradient = (y[num_entries*4//5] - y[num_entries-1]) / (x[num_entries*4//5] - x[num_entries-1])
    return np.arctan(-gradient)

def calculate_initial_conditions(b, initial_x):
    initial_r = np.sqrt(b**2 + initial_x**2)
    initial_phi = np.arccos(initial_x / initial_r)
    initial_rdot = np.cos(initial_phi)
    initial_phidot = -np.sqrt((1 - initial_rdot**2) / initial_r**2)
    L = initial_r**2 * initial_phidot
    return [initial_r, initial_rdot, initial_phi], L


def generate_trajectory_figure(bs, M, initial_x, max_t, t_eval):
    fig = go.Figure()
    R = 2*M
    for b in bs:
        initial, L = calculate_initial_conditions(b, initial_x)
        r, phi = solve((0, max_t), t_eval, initial, [M, L])
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'b={b}'))

    circle = go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=20, color='black'), name='Black Hole')
    fig.add_trace(circle)

    fig.update_layout(
        title='Light rays around a Schwarzschild black hole',
        xaxis_title='x (coordinate)',
        yaxis_title='y (coordinate)',
        yaxis_scaleanchor='x',
        height=600
    )
    return fig

def generate_deflection_figure(bs, M, initial_x, max_t, t_eval):
    deflections = []
    for b in bs:
        initial, L = calculate_initial_conditions(b, initial_x)
        r, phi = solve((0, max_t), t_eval, initial, [M, L])
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        deflections.append(calculate_deflection(x, y, t_eval))

    expected = expected_schw_light_bending(bs, M)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bs, y=expected, mode='markers', name='Theoretical deflection'))
    fig.add_trace(go.Scatter(x=bs, y=deflections, mode='markers+lines', name='Numerical result'))

    fig.update_layout(
        title='Deflection angle vs Impact parameter',
        xaxis_title='Impact parameter',
        yaxis_title='Deflection angle (radians)',
        height=600
    )
    return fig


app = dash.Dash(__name__)
app.title = "Schwarzschild Light Bending"

app.layout = html.Div([
    html.H1("Light Bending around a Schwarzschild Black Hole"),

    html.Label("Black Hole Mass (M):"),
    dcc.Slider(id='mass-slider', min=0.5, max=5.0, step=0.1, value=1.0, marks={i: str(i) for i in range(1, 6)}),

    dcc.Graph(id='trajectory-plot'),
    dcc.Graph(id='deflection-plot')
])

@app.callback(
    [Output('trajectory-plot', 'figure'),
     Output('deflection-plot', 'figure')],
    [Input('mass-slider', 'value')]
)
def update_plots(M):
    initial_x = -50
    max_t = 10000.
    t_eval = np.arange(0, max_t, 0.01)
    bs1 = np.array([4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40])
    bs2 = np.linspace(20, 100, 40)
    traj_fig = generate_trajectory_figure(bs1, M, initial_x, max_t, t_eval)
    defl_fig = generate_deflection_figure(bs2, M, initial_x, max_t, t_eval)
    return traj_fig, defl_fig


if __name__ == '__main__':
    app.run(debug=True)
