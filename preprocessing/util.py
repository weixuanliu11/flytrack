import numpy as np

def calculate_curvature(trajectory, dt=0.04):
    trajectory = np.array(trajectory)
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    dx = np.diff(x)/dt
    dy = np.diff(y)/dt

    # Create shift arrays for second derivatives
    dx_shift =  np.roll(dx, -1)/dt
    dy_shift = np.roll(dy, -1)/dt

    # Calculate curvature
    curvature = (dx * dy_shift - dy * dx_shift) / (dx**2 + dy**2)**(3/2)

    return list(curvature)

# Calculate the curvature of a list of trajectories
def traj_curve(trajectories):
  curv = []
  for traj in trajectories:
    cv_traj = calculate_curvature(traj)
    curv_traj = np.array(cv_traj + [cv_traj[-1]]).reshape(-1, 1)
    curv.append(curv_traj)
  curv = np.vstack(curv)
  return curv

def calculate_egocentric_angle(v1, v2):
    # Represent v1 and v2 as complex numbers
    v1_complex = v1[0] + 1j * v1[1]
    v2_complex = v2[0] + 1j * v2[1]

    # Calculate the relative angle from v1 to v2
    relative_angle = np.angle(v2_complex / v1_complex, deg=False)  # result in radians

    return relative_angle

# Calculate instantaneous speed and turn from trajectory(absolute position)
def allo_agent_speed_turn(trajectory, dt=0.04):
  trajectory = np.array(trajectory)
  x = trajectory[:, 0]
  y = trajectory[:, 1]
  dx = np.diff(x)/dt
  dy = np.diff(y)/dt

  # Create vectors from (dx, dy)
  vectors = list(zip(dx, dy))
  egocentric_dangles = [calculate_egocentric_angle(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]

  turn = [egocentric_dangles[0], egocentric_dangles[0]] + list(egocentric_dangles) # /dt
  speed = np.sqrt(dx**2 + dy**2)
  speed = [speed[0]] + list(speed)

  return np.array(speed), np.array(turn)

# Calculate the allocenter agent speed and turn of a list of trajectories
def traj_speed_turn(trajectories):
  combined_data = []
  for traj in trajectories:
    speed, turn = allo_agent_speed_turn(traj)
    combined_data.append(np.column_stack((speed, turn)))
  result = np.vstack(combined_data)
  return result

