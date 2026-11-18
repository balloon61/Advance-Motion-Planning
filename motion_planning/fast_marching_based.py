import numpy as np
import matplotlib.pyplot as plt
import heapq

# ==========================================================
# PARAMETERS
# ==========================================================

RES = 0.5

MAP_WIDTH = 70.0
MAP_HEIGHT = 60.0
NX = int(MAP_WIDTH / RES)
NY = int(MAP_HEIGHT / RES)

NTH = 32

INF = 1e9

DT = 0.1

# ==========================================================
# VEHICLE
# ==========================================================

MASS = 20.0
IZ = 8.0

DAMP_U = 0.5
DAMP_R = 1.0

ARM = 0.5

# ==========================================================
# GOAL
# ==========================================================

GOAL_X = int(55 / RES)
GOAL_Y = int(20 / RES)
GOAL_HEADING = np.pi / 2

# ==========================================================
# COSTS
# ==========================================================

GEAR_SWITCH_COST = 3.0
REVERSE_COST = 0.2
TURN_COST = 2.0

# ==========================================================
# CONTROLS
# ==========================================================

CONTROLS = [
    (15.0, 0.0),
    (20.0, 0.0),
    (20.0, 0.2),
    (20.0, -0.2),
    (20.0, 0.4),
    (20.0, -0.4),
    (25.0, 0.7),
    (25.0, -0.7),
]

GEARS = [1, -1]

# ==========================================================
# DISTURBANCE
# ==========================================================

def disturbance():
    """Return the external force in world-frame newtons."""
    return np.array([
        -10.0,
        10.0,
    ])

# ==========================================================
# ANGLES
# ==========================================================

def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def idx_to_theta(index):
    return index * 2 * np.pi / NTH


def theta_to_idx(theta):
    theta = wrap(theta)
    if theta < 0:
        theta += 2 * np.pi
    return int(round(theta * NTH / (2 * np.pi))) % NTH

# ==========================================================
# DYNAMIC ROLLOUT
# ==========================================================

def rollout(theta, gear, thrust, steer, duration=1.5):
    x = 0.0
    y = 0.0
    u = 0.0
    r = 0.0
    drift_x = 0.0
    drift_y = 0.0
    force_x, force_y = disturbance()
    th = theta
    cost = 0.0
    t = 0.0

    while t < duration:
        F = gear * thrust
        udot = F / MASS - DAMP_U * u
        torque = F * np.sin(steer) * ARM
        rdot = torque / IZ - DAMP_R * r

        u += udot * DT
        r += rdot * DT
        drift_x += (force_x / MASS - DAMP_U * drift_x) * DT
        drift_y += (force_y / MASS - DAMP_U * drift_y) * DT
        th += r * DT
        x += (u * np.cos(th) + drift_x) * DT
        y += (u * np.sin(th) + drift_y) * DT
        cost += DT + 0.0005 * thrust**2 + 0.05 * r**2
        t += DT

    return x, y, wrap(th), cost

# ==========================================================
# PRECOMPUTE PRIMITIVES
# ==========================================================

inverse_primitive_table = {
    gear: {heading: [] for heading in range(NTH)} for gear in GEARS
}
for prev_ith in range(NTH):
    theta = idx_to_theta(prev_ith)
    for gear in GEARS:
        for thrust, steer in CONTROLS:
            dx, dy, end_theta, edge_cost = rollout(theta, gear, thrust, steer)
            end_ith = theta_to_idx(end_theta)
            inverse_primitive_table[gear][end_ith].append(
                (prev_ith, dx, dy, end_theta, edge_cost)
            )

# ==========================================================
# VALUE FUNCTION
# ==========================================================

V = np.ones((NX, NY, NTH, 2)) * INF
pq = []
parent = {}

# ==========================================================
# GOAL SET
# ==========================================================

goal_states = set()
goal_heading_indices = [
    theta_to_idx(GOAL_HEADING + (np.pi if gear == -1 else 0))
    for gear in GEARS
]
for gear_idx, heading_idx in enumerate(goal_heading_indices):
    goal = (GOAL_X, GOAL_Y, heading_idx, gear_idx)
    goal_states.add(goal)
    V[goal] = 0
    heapq.heappush(pq, (0.0, *goal))

# ==========================================================
# BACKWARD SEARCH
# ==========================================================

while pq:
    c, ix, iy, ith, igear = heapq.heappop(pq)
    if c > V[ix, iy, ith, igear]:
        continue

    current_gear = GEARS[igear]
    for prev_gear_idx, prev_gear in enumerate(GEARS):
        for prev_ith, dx, dy, end_theta, edge_cost in (
            inverse_primitive_table[current_gear][ith]
        ):
            px = int(round(ix - dx / RES))
            py = int(round(iy - dy / RES))
            pth = prev_ith

            if not 0 <= px < NX:
                continue
            if not 0 <= py < NY:
                continue

            switch_cost = 0.0
            if prev_gear_idx != igear:
                switch_cost = GEAR_SWITCH_COST
            reverse_cost = REVERSE_COST if prev_gear == -1 else 0.0
            heading_change = wrap(end_theta - idx_to_theta(prev_ith))
            turn_cost = TURN_COST * heading_change**2
            new_cost = (
                c
                + edge_cost
                + switch_cost
                + reverse_cost
                + turn_cost
            )

            if new_cost < V[px, py, pth, prev_gear_idx]:
                V[px, py, pth, prev_gear_idx] = new_cost
                parent[(px, py, pth, prev_gear_idx)] = (ix, iy, ith, igear)
                heapq.heappush(
                    pq, (new_cost, px, py, pth, prev_gear_idx)
                )

# ==========================================================
# START
# ==========================================================

start = (int(5 / RES), int(10 / RES), theta_to_idx(0), 0)

# ==========================================================
# PATH
# ==========================================================

xs = []
ys = []
ths = []
node = start
count = 0
while node in parent and count < 300:
    x, y, th, g = node
    xs.append(x)
    ys.append(y)
    ths.append(idx_to_theta(th))
    node = parent[node]
    count += 1

if node not in goal_states:
    raise RuntimeError(f"No path to the goal pose found; stopped at {node}")

xs.append(node[0])
ys.append(node[1])
ths.append(idx_to_theta(node[2]))

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(10, 8))
xx = np.arange(0, NX, 4) * RES
yy = np.arange(0, NY, 4) * RES
X, Y = np.meshgrid(xx, yy)
U = np.zeros_like(X, dtype=float)
W = np.zeros_like(Y, dtype=float)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        d = disturbance()
        U[i, j] = d[0]
        W[i, j] = d[1]

plt.quiver(X, Y, U, W, alpha=0.4)
plt.scatter(GOAL_X * RES, GOAL_Y * RES, c="red", s=80, label="Goal")
for gear_idx, color in enumerate(("red", "darkorange")):
    target_heading = idx_to_theta(goal_heading_indices[gear_idx])
    plt.arrow(
        GOAL_X * RES,
        GOAL_Y * RES,
        2.0 * np.cos(target_heading),
        2.0 * np.sin(target_heading),
        width=0.08,
        color=color,
        label=f"Goal heading ({'forward' if GEARS[gear_idx] == 1 else 'reverse'})",
    )
plt.plot(np.array(xs) * RES, np.array(ys) * RES, "b", linewidth=3)
for k in range(0, len(xs), 2):
    plt.arrow(
        xs[k] * RES,
        ys[k] * RES,
        1.3 * np.cos(ths[k]),
        1.3 * np.sin(ths[k]),
        width=0.05,
        color="blue",
    )

plt.scatter(start[0] * RES, start[1] * RES, c="green", s=80)
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title(
    "Fast Marching Style Dock Planner\nForward + Reverse + Disturbance"
)
plt.show()