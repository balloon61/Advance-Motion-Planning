import numpy as np

# =====================================================
# CONFIG
# =====================================================

DT = 0.1

HORIZON = 40
ROLLOUTS = 1000

LAMBDA = 15.0

MAX_THRUST = 30.0
MAX_THRUST_ANGLE = np.deg2rad(45)

MASS = 50.0
INERTIA_Z = 10.0

DAMP_X = 1.0
DAMP_Y = 1.0
DAMP_YAW = 2.0

THRUSTER_OFFSET = 0.5

MAX_SPEED = 1.0
MAX_ACCELERATION = 0.3
MAX_YAW_RATE = np.deg2rad(30.0)

# =====================================================
# DISTURBANCE FIELD
# =====================================================

def disturbance_field(x, y):

    current_x = 0.8

    current_y = 0.4 * np.sin(y * 0.1)

    return np.array([
        current_x,
        current_y
    ])

# =====================================================
# DYNAMICS
# =====================================================

def propagate(state, control):

    x, y, yaw, vx, vy, r = state

    thrust = control[0]
    delta = control[1]

    thrust = np.clip(
        thrust,
        -MAX_THRUST,
        MAX_THRUST
    )

    delta = np.clip(
        delta,
        -MAX_THRUST_ANGLE,
        MAX_THRUST_ANGLE
    )

    current = disturbance_field(x, y)

    fx = thrust * np.cos(yaw + delta)
    fy = thrust * np.sin(yaw + delta)

    ax = fx / MASS - DAMP_X * vx
    ay = fy / MASS - DAMP_Y * vy

    yaw_torque = (
        thrust *
        np.sin(delta) *
        THRUSTER_OFFSET
    )

    r_dot = (
        yaw_torque / INERTIA_Z
        -
        DAMP_YAW * r
    )

    vx += ax * DT
    vy += ay * DT

    r += r_dot * DT

    world_vx = (
        np.cos(yaw) * vx
        -
        np.sin(yaw) * vy
    )

    world_vy = (
        np.sin(yaw) * vx
        +
        np.cos(yaw) * vy
    )

    world_vx += current[0]
    world_vy += current[1]

    x += world_vx * DT
    y += world_vy * DT

    yaw += r * DT

    yaw = np.arctan2(
        np.sin(yaw),
        np.cos(yaw)
    )

    return np.array([
        x,
        y,
        yaw,
        vx,
        vy,
        r
    ])


def propagate_batch(states, controls):

    x, y, yaw, vx, vy, r = states.T

    thrust = np.clip(
        controls[:, 0],
        -MAX_THRUST,
        MAX_THRUST
    )

    delta = np.clip(
        controls[:, 1],
        -MAX_THRUST_ANGLE,
        MAX_THRUST_ANGLE
    )

    current_y = 0.4 * np.sin(y * 0.1)

    fx = thrust * np.cos(yaw + delta)
    fy = thrust * np.sin(yaw + delta)

    ax = fx / MASS - DAMP_X * vx
    ay = fy / MASS - DAMP_Y * vy

    yaw_torque = thrust * np.sin(delta) * THRUSTER_OFFSET
    r_dot = yaw_torque / INERTIA_Z - DAMP_YAW * r

    vx = vx + ax * DT
    vy = vy + ay * DT
    r = r + r_dot * DT

    world_vx = np.cos(yaw) * vx - np.sin(yaw) * vy + 0.8
    world_vy = np.sin(yaw) * vx + np.cos(yaw) * vy + current_y

    x = x + world_vx * DT
    y = y + world_vy * DT
    yaw = yaw + r * DT
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))

    return np.column_stack((x, y, yaw, vx, vy, r))


def generate_global_curve(
    start,
    goal,
    start_yaw,
    goal_yaw,
    max_speed=MAX_SPEED,
    max_acceleration=MAX_ACCELERATION,
    max_yaw_rate=MAX_YAW_RATE,
    dt=DT
):

    if max_speed <= 0.0 or max_acceleration <= 0.0 or max_yaw_rate <= 0.0 or dt <= 0.0:
        raise ValueError('Path limits and time step must be positive')

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    chord = np.linalg.norm(goal - start)

    if chord == 0.0:
        raise ValueError('Start and goal positions must differ')

    handle = chord / 3.0
    start_handle = start + handle * np.array([
        np.cos(start_yaw),
        np.sin(start_yaw)
    ])
    goal_handle = goal - handle * np.array([
        np.cos(goal_yaw),
        np.sin(goal_yaw)
    ])

    parameter = np.linspace(0.0, 1.0, 2001)
    inverse = 1.0 - parameter

    curve = (
        inverse[:, None]**3 * start
        +
        3.0 * inverse[:, None]**2 * parameter[:, None] * start_handle
        +
        3.0 * inverse[:, None] * parameter[:, None]**2 * goal_handle
        +
        parameter[:, None]**3 * goal
    )

    first_derivative = (
        3.0 * inverse[:, None]**2 * (start_handle - start)
        +
        6.0 * inverse[:, None] * parameter[:, None] * (goal_handle - start_handle)
        +
        3.0 * parameter[:, None]**2 * (goal - goal_handle)
    )

    second_derivative = (
        6.0 * inverse[:, None] * (goal_handle - 2.0 * start_handle + start)
        +
        6.0 * parameter[:, None] * (goal - 2.0 * goal_handle + start_handle)
    )

    arc_length = np.concatenate((
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
    ))

    dx, dy = first_derivative.T
    ddx, ddy = second_derivative.T
    denominator = np.maximum(dx**2 + dy**2, 1e-12)
    curvature = (dx * ddy - dy * ddx) / denominator**1.5
    yaw = np.unwrap(np.arctan2(dy, dx))

    lateral_acceleration_limit = max_acceleration / np.sqrt(2.0)
    longitudinal_acceleration_limit = max_acceleration / np.sqrt(2.0)
    curvature_magnitude = np.abs(curvature)
    speed_limit = np.minimum(
        max_speed,
        np.minimum(
            max_yaw_rate / np.maximum(curvature_magnitude, 1e-12),
            np.sqrt(
                lateral_acceleration_limit
                / np.maximum(curvature_magnitude, 1e-12)
            )
        )
    )
    speed_limit[[0, -1]] = 0.0

    speed = speed_limit.copy()
    for index in range(1, len(speed)):
        distance = arc_length[index] - arc_length[index - 1]
        speed[index] = min(
            speed[index],
            np.sqrt(
                speed[index - 1]**2
                +
                2.0 * longitudinal_acceleration_limit * distance
            )
        )

    for index in range(len(speed) - 2, -1, -1):
        distance = arc_length[index + 1] - arc_length[index]
        speed[index] = min(
            speed[index],
            np.sqrt(
                speed[index + 1]**2
                +
                2.0 * longitudinal_acceleration_limit * distance
            )
        )

    segment_duration = (
        2.0 * np.diff(arc_length)
        /
        np.maximum(speed[:-1] + speed[1:], 1e-12)
    )
    profile_time = np.concatenate((
        [0.0],
        np.cumsum(segment_duration)
    ))

    sample_time = np.arange(0.0, profile_time[-1], dt)
    sample_time = np.append(sample_time, profile_time[-1])
    sample_distance = np.interp(sample_time, profile_time, arc_length)

    sample_x = np.interp(sample_distance, arc_length, curve[:, 0])
    sample_y = np.interp(sample_distance, arc_length, curve[:, 1])
    sample_yaw = np.interp(sample_distance, arc_length, yaw)
    sample_speed = np.interp(sample_time, profile_time, speed)
    sample_curvature = np.interp(sample_distance, arc_length, curvature)
    sample_yaw_rate = sample_curvature * sample_speed

    return np.column_stack((
        sample_time,
        sample_x,
        sample_y,
        sample_yaw,
        sample_speed,
        sample_yaw_rate
    ))


# =====================================================
# GEOMETRY
# =====================================================

def point_segment_distance(
    point,
    a,
    b
):

    ab = b - a

    t = np.dot(
        point - a,
        ab
    ) / np.dot(ab, ab)

    t = np.clip(t, 0.0, 1.0)

    closest = a + t * ab

    dist = np.linalg.norm(
        point - closest
    )

    return dist


def point_segment_distance_batch(points, a, b):

    ab = b - a

    t = np.sum(
        (points - a) * ab,
        axis=1
    ) / np.dot(ab, ab)

    t = np.clip(t, 0.0, 1.0)

    closest = a + t[:, None] * ab

    return np.linalg.norm(
        points - closest,
        axis=1
    )

# =====================================================
# COST
# =====================================================

def wrap_pi(angle):

    return np.arctan2(
        np.sin(angle),
        np.cos(angle)
    )

def cost_function(
    state,
    control,
    seg_a,
    seg_b,
    goal_yaw
):

    pos = state[:2]

    dist_line = point_segment_distance(
        pos,
        seg_a,
        seg_b
    )

    yaw_error = wrap_pi(
        state[2] - goal_yaw
    )

    body_speed = np.sqrt(
        state[3]**2 +
        state[4]**2
    )

    thrust = control[0]

    cost = (
        10.0 * dist_line
        +
        3.0 * yaw_error**2
        +
        0.02 * body_speed**2
        +
        0.001 * thrust**2
    )

    return cost


def cost_function_batch(
    states,
    controls,
    seg_a,
    seg_b,
    goal_yaw
):

    dist_line = point_segment_distance_batch(
        states[:, :2],
        seg_a,
        seg_b
    )

    yaw_error = wrap_pi(
        states[:, 2] - goal_yaw
    )

    body_speed = np.sqrt(
        states[:, 3]**2 +
        states[:, 4]**2
    )

    thrust = controls[:, 0]

    return (
        10.0 * dist_line
        +
        3.0 * yaw_error**2
        +
        0.02 * body_speed**2
        +
        0.001 * thrust**2
    )


def terminal_cost(
    state,
    seg_a,
    seg_b,
    goal_yaw
):

    pos = state[:2]

    dist = point_segment_distance(
        pos,
        seg_a,
        seg_b
    )

    yaw_error = wrap_pi(
        state[2] - goal_yaw
    )

    speed = np.sqrt(
        state[3]**2 +
        state[4]**2
    )

    return (
        300.0 * dist
        +
        100.0 * yaw_error**2
        +
        5.0 * speed**2
    )


def terminal_cost_batch(
    states,
    seg_a,
    seg_b,
    goal_yaw
):

    dist = point_segment_distance_batch(
        states[:, :2],
        seg_a,
        seg_b
    )

    yaw_error = wrap_pi(
        states[:, 2] - goal_yaw
    )

    speed = np.sqrt(
        states[:, 3]**2 +
        states[:, 4]**2
    )

    return (
        300.0 * dist
        +
        100.0 * yaw_error**2
        +
        5.0 * speed**2
    )


def path_speed_batch(states, path_yaw):

    current_y = 0.4 * np.sin(states[:, 1] * 0.1)
    world_vx = (
        np.cos(states[:, 2]) * states[:, 3]
        -
        np.sin(states[:, 2]) * states[:, 4]
        +
        0.8
    )
    world_vy = (
        np.sin(states[:, 2]) * states[:, 3]
        +
        np.cos(states[:, 2]) * states[:, 4]
        +
        current_y
    )

    return (
        world_vx * np.cos(path_yaw)
        +
        world_vy * np.sin(path_yaw)
    )


def trajectory_cost_batch(states, controls, reference):

    position_error = np.linalg.norm(
        states[:, :2] - reference[:2],
        axis=1
    )

    yaw_error = wrap_pi(
        states[:, 2] - reference[2]
    )

    speed_error = path_speed_batch(states, reference[2]) - reference[3]
    yaw_rate_error = states[:, 5] - reference[4]
    thrust = controls[:, 0]

    return (
        10.0 * position_error
        +
        3.0 * yaw_error**2
        +
        3.0 * speed_error**2
        +
        0.5 * yaw_rate_error**2
        +
        0.001 * thrust**2
    )


def trajectory_terminal_cost_batch(states, reference):

    position_error = np.linalg.norm(
        states[:, :2] - reference[:2],
        axis=1
    )

    yaw_error = wrap_pi(
        states[:, 2] - reference[2]
    )

    speed_error = path_speed_batch(states, reference[2]) - reference[3]
    yaw_rate_error = states[:, 5] - reference[4]

    return (
        300.0 * position_error
        +
        100.0 * yaw_error**2
        +
        20.0 * speed_error**2
        +
        5.0 * yaw_rate_error**2
    )


# =====================================================
# MPPI
# =====================================================

class MPPI:

    def __init__(self):

        self.u = np.zeros(
            (
                HORIZON,
                2
            )
        )

        self.sigma = np.array([
            8.0,
            np.deg2rad(8)
        ])

    def command(
        self,
        state,
        global_path,
        elapsed_time
    ):

        reference_times = (
            elapsed_time
            +
            np.arange(1, HORIZON + 1) * DT
        )
        references = np.column_stack([
            np.interp(reference_times, global_path[:, 0], global_path[:, column])
            for column in range(1, 6)
        ])

        noise_bank = (
            np.random.randn(
                ROLLOUTS,
                HORIZON,
                2
            )
            *
            self.sigma
        )

        controls = self.u[None, :, :] + noise_bank
        states = np.broadcast_to(
            state,
            (ROLLOUTS, state.size)
        ).copy()
        costs = np.zeros(ROLLOUTS)

        for t in range(HORIZON):

            states = propagate_batch(
                states,
                controls[:, t, :]
            )

            costs += trajectory_cost_batch(
                states,
                controls[:, t, :],
                references[t]
            )

        costs += trajectory_terminal_cost_batch(
            states,
            references[-1]
        )

        beta = np.min(costs)

        weights = np.exp(
            -(costs - beta)
            / LAMBDA
        )

        weights /= np.sum(weights)

        self.u += np.sum(
            noise_bank * weights[:, None, None],
            axis=0
        )

        control = self.u[0].copy()

        self.u[:-1] = self.u[1:]
        self.u[-1] = self.u[-2]

        return control


# =====================================================
# DEMO
# =====================================================

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    controller = MPPI()

    state = np.array([
        0.0,    # x
        0.0,    # y
        0.0,    # yaw
        0.0,    # vx
        0.0,    # vy
        0.0     # yaw rate
    ])

    seg_a = np.array([
        22.0,
        5.0
    ])

    seg_b = np.array([
        22.0,
        15.0
    ])

    goal_yaw = np.pi/2

    planned_path = generate_global_curve(
        state[:2],
        seg_b,
        state[2],
        goal_yaw
    )
    print(
        f'Planned {len(planned_path)} path poses '
        f'over {planned_path[-1, 0]:.1f} seconds'
    )

    state_history = [state.copy()]
    control_history = []

    steps = int(np.ceil(planned_path[-1, 0] / DT))
    for step in range(steps):

        u = controller.command(
            state,
            planned_path,
            step * DT
        )

        state = propagate(
            state,
            u
        )

        state_history.append(state.copy())
        control_history.append(u.copy())

        print(
            f'{step:03d}',
            f'x={state[0]:.2f}',
            f' y={state[1]:.2f}',
            f' yaw={np.rad2deg(state[2]):.1f}',
            f' thrust={u[0]:.2f}',
            f' delta={np.rad2deg(u[1]):.1f}'
        )

    states = np.array(state_history)
    controls = np.array(control_history)
    state_time = np.arange(len(states)) * DT
    control_time = state_time[:-1]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].plot(states[:, 0], states[:, 1], label='Trajectory')
    axes[0, 0].plot(
        [seg_a[0], seg_b[0]],
        [seg_a[1], seg_b[1]],
        'r--',
        label='Goal segment'
    )
    axes[0, 0].plot(
        planned_path[:, 1],
        planned_path[:, 2],
        'k:',
        label='Global path'
    )
    axes[0, 0].scatter(states[0, 0], states[0, 1], label='Start')
    axes[0, 0].scatter(states[-1, 0], states[-1, 1], label='End')
    axes[0, 0].set_title('Path')
    axes[0, 0].set_xlabel('x (m)')
    axes[0, 0].set_ylabel('y (m)')
    axes[0, 0].legend()

    axes[0, 1].plot(state_time, states[:, 0], label='x')
    axes[0, 1].plot(state_time, states[:, 1], label='y')
    axes[0, 1].set_title('Position')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Position (m)')
    axes[0, 1].legend()

    axes[0, 2].plot(
        state_time,
        np.rad2deg(states[:, 2]),
        label='Yaw'
    )
    axes[0, 2].plot(
        planned_path[:, 0],
        np.rad2deg(planned_path[:, 3]),
        'k:',
        label='Planned yaw'
    )
    axes[0, 2].axhline(
        np.rad2deg(goal_yaw),
        color='r',
        linestyle='--',
        label='Goal yaw'
    )
    axes[0, 2].set_title('Heading')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Angle (deg)')
    axes[0, 2].legend()

    axes[1, 0].plot(state_time, states[:, 3], label='vx')
    axes[1, 0].plot(state_time, states[:, 4], label='vy')
    world_vx = (
        np.cos(states[:, 2]) * states[:, 3]
        -
        np.sin(states[:, 2]) * states[:, 4]
        +
        0.8
    )
    world_vy = (
        np.sin(states[:, 2]) * states[:, 3]
        +
        np.cos(states[:, 2]) * states[:, 4]
        +
        0.4 * np.sin(states[:, 1] * 0.1)
    )
    axes[1, 0].plot(
        state_time,
        np.hypot(world_vx, world_vy),
        label='Ground speed'
    )
    axes[1, 0].plot(
        state_time,
        np.interp(state_time, planned_path[:, 0], planned_path[:, 4]),
        'k:',
        label='Planned speed'
    )
    axes[1, 0].set_title('Body velocity')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Velocity (m/s)')
    axes[1, 0].legend()

    axes[1, 1].plot(
        state_time,
        np.rad2deg(states[:, 5]),
        label='Yaw rate'
    )
    axes[1, 1].plot(
        planned_path[:, 0],
        np.rad2deg(planned_path[:, 5]),
        'k:',
        label='Planned yaw rate'
    )
    axes[1, 1].set_title('Yaw rate')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Rate (deg/s)')
    axes[1, 1].legend()

    control_axis = axes[1, 2]
    control_axis.plot(control_time, controls[:, 0], label='Thrust')
    control_axis.set_title('Controls')
    control_axis.set_xlabel('Time (s)')
    control_axis.set_ylabel('Thrust')

    angle_axis = control_axis.twinx()
    angle_axis.plot(
        control_time,
        np.rad2deg(controls[:, 1]),
        color='tab:orange',
        label='Thrust angle'
    )
    angle_axis.set_ylabel('Angle (deg)')
    control_lines, control_labels = control_axis.get_legend_handles_labels()
    angle_lines, angle_labels = angle_axis.get_legend_handles_labels()
    control_axis.legend(
        control_lines + angle_lines,
        control_labels + angle_labels
    )

    for axis in axes.flat:
        axis.grid(True)

    fig.tight_layout()
    plt.show()
