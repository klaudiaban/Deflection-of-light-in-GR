import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.integrate import solve_ivp

sns.set(style="darkgrid")

with open("config.json") as f:
    config = json.load(f)

METHOD = config["method"]

def schw_null_geodesics(t, w, M, L):
    r, rdot, phi = w
    phidot = L / r**2
    rddot = L**2 * (r - 3*M) / r**4
    return np.array([rdot, rddot, phidot])

def rk4_step(f, t, w, h, *args):
    k1 = f(t, w, *args)
    k2 = f(t + h/2, w + h*k1/2, *args)
    k3 = f(t + h/2, w + h*k2/2, *args)
    k4 = f(t + h, w + h*k3, *args)
    return w + h * (k1 + 2*k2 + 2*k3 + k4) / 6

def euler_step(f, t, w, h, *args):
    return w + h * f(t, w, *args)

def solve(t_span, t_eval, initial, p):
    M, L = p
    R = 2 * M
    h = t_eval[1] - t_eval[0]
    n = len(t_eval)

    if METHOD == "solve_ivp":
        sol = solve_ivp(schw_null_geodesics, t_span, initial, t_eval=t_eval, args=(M, L), rtol=1e-9, atol=1e-9)
        r = sol.y[0]
        phi = sol.y[2]
        return r, phi

    w = np.zeros((3, n))
    w[:, 0] = initial
    last_index = n
    step_func = rk4_step if METHOD == "rk4" else euler_step

    for i in range(1, n):
        w[:, i] = step_func(schw_null_geodesics, t_eval[i-1], w[:, i-1], h, M, L)
        if w[0, i] <= R:
            last_index = i + 1
            break

    r = w[0, :last_index]
    phi = w[2, :last_index]
    return r, phi

def expected_schw_light_bending(r, M):
    return 4. * M / r

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

def plot_both():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    M = 1.
    R = 2*M
    initial_x = -50
    max_t = 1000.
    t_eval = np.arange(0, max_t, 0.05)

    # --- Plot 1: Light rays ---
    bs1 = np.array([4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40])
    for b in bs1:
        initial, L = calculate_initial_conditions(b, initial_x)
        r, phi = solve((0, max_t), t_eval, initial, [M, L])
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        ax1.plot(x, y, label=f"b={b}", linestyle='-', linewidth=2)

    lim = -initial_x
    ax1.set_xlim([-lim, lim])
    ax1.set_ylim([-lim, lim])
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title(f'Light rays around a Schwarzschild black hole ({METHOD})', fontsize=14)
    ax1.set_xlabel('x (coordinate)', fontsize=12)
    ax1.set_ylabel('y (coordinate)', fontsize=12)
    ax1.legend(title="Impact parameter", loc='upper right', fontsize=10)

    circle = plt.Circle((0., 0.), R, color='black', fill=True, zorder=10)
    ax1.add_artist(circle)

    # --- Plot 2: Deflection angles ---
    bs2 = np.linspace(20, 100, 40)
    deflections = []
    for b in bs2:
        initial, L = calculate_initial_conditions(b, initial_x)
        r, phi = solve((0, max_t), t_eval, initial, [M, L])
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        deflections.append(calculate_deflection(x, y, t_eval[:len(r)]))

    expected = expected_schw_light_bending(bs2, M)
    ax2.plot(bs2, expected, 'ro', label=r'Theoretical deflection ($\frac{4M}{R}$)', markersize=6)
    ax2.plot(bs2, deflections, 'b+', label=f'Numerical result ({METHOD})', markersize=8)
    ax2.set_title('Deflection angle vs Impact parameter', fontsize=14)
    ax2.set_xlabel('Impact parameter', fontsize=12)
    ax2.set_ylabel('Deflection angle (radians)', fontsize=12)
    ax2.legend(fontsize=10)

    ax2.grid(True)
    ax2.set_xlim([0, max(bs2)+5])
    ax2.set_ylim([0, max(deflections)+0.05])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.show()

if __name__ == '__main__':
    plot_both()
