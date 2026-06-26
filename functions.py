# helper function for the dynamical emulator
# lorenze 63 ground truth
def lorenze_63(t ,state ,args = parms):
  x, y, z = state
  sigma, row, beta = args

  dx = sigma*(y - x) # intesty of convection
  dy = x*(row - z) - y # temperature diffrence
  dz = x*y - beta*z # distortion of the temperature profile

  return jnp.array([dx, dy, dz])

# stepper
def stepper(carry, x):
  solution = diffeqsolve(term, solver, t0 = 0, t1 = dt, dt0 = dt, y0 = carry, args = parms)
  next_carry = solution.ys[-1]
  return next_carry ,next_carry

# lyaponove exponent stepper
def lyaponove_stepper(carry, _):
    state_A, state_B = carry

    sol_A = diffeqsolve(term, solver, t0 = 0, t1 = dt, dt0 = dt, y0 = state_A, args = parms)

    sol_B = diffeqsolve(term, solver, t0 = 0, t1 = dt, dt0 = dt, y0 = state_B, args = parms)

    next_state_A = sol_A.ys[0]
    next_state_B = sol_B.ys[0]

    current_distance = jnp.linalg.norm(next_state_B - next_state_A)

    growth_factor = current_distance / delta

    direction_vector = (next_state_B - next_state_A) / current_distance
    adjusted_state_B = next_state_A + (direction_vector * delta)

    next_carry = (next_state_A, adjusted_state_B)

    return next_carry, (next_state_A, growth_factor)

def generate_singel_training_traj(y0):
  save_ts_gen = jnp.arange(0.0, total_steps * dt, dt)
  sol = diffeqsolve(term, solver, t0 = 0, t1 = total_steps * dt, dt0 = dt, y0 = y0, args = parms, saveat=SaveAt(ts=save_ts_gen), max_steps=total_steps * 10)
  full_path = sol.ys.reshape(total_steps, 3)
  burnt_in_path = full_path[burn_in_steps:]
  return burnt_in_path

# creating training batches
def training_dataset(trajectory, horizon, stride=1):
    window_length = horizon + 1

    # Calculate max_start_idx based on the actual length of the provided trajectory
    max_start_idx = trajectory.shape[0] - window_length
    start_indices = jnp.arange(start=0, stop=max_start_idx, step=stride)

    def extract_single_window(start_idx):
        single_slice = jax.lax.dynamic_slice(trajectory, (start_idx, 0), (window_length, 3))
        return single_slice

    all_windows = jax.vmap(extract_single_window)(start_indices)

    X_data = all_windows[:, 0, :].astype(jnp.float64)
    Y_data = all_windows[:, 1:, :].astype(jnp.float64)

    return X_data, Y_data

# rollout func
def prediction_func(model, init_state, horizontal_lenght):
    def scan_stepper(carry, _):
        current_state = carry
        next_state = model(current_state)
        return next_state, next_state
    _, prediction_seq = jax.lax.scan(scan_stepper, init_state, jnp.arange(horizontal_lenght))
    return prediction_seq
    
# loss and training stepper
def loss_func(model, x_single, y_trajectory):
    # x_single is (state_dim), y_trajectory is (horizon, state_dim)
    horizon = y_trajectory.shape[0]
    prediction_trajectory = prediction_func(model, x_single, horizon)
    loss = jnp.mean((prediction_trajectory - y_trajectory)**2)
    return loss

def batch_step_loss_func(model, x_batch, y_batch):
    # x_batch is (batch_size, state_dim), y_batch is (batch_size, horizon, state_dim)
    loss_sample = jax.vmap(loss_func, in_axes=(None, 0, 0))(model, x_batch, y_batch)
    return jnp.mean(loss_sample)

@eqx.filter_jit
def model_stepper(model, opt_state, optimizer, x_batch, y_batch):
    loss, grads = eqx.filter_value_and_grad(batch_step_loss_func)(model, x_batch, y_batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


def save_model(model, filename="lorenz_mlp.eqx", folder="saved_models"):
    dir_path = Path(folder)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    file_path = dir_path / filename
    eqx.tree_serialise_leaves(file_path, model)
    print(f" Saved model to: {file_path}")

def load_model(model_skeleton, filename="lorenz_mlp.eqx", folder="saved_models"):
    file_path = Path(folder) / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"No saved checkpoint at {file_path}")
        
    loaded_model = eqx.tree_deserialise_leaves(file_path, model_skeleton)
    print(f" Loaded model from: {file_path}")
    return loaded_model

# fractal dimentions
def calculate_fractal_dimension(trajectory, r_min=0.1, r_max=10.0, num_r=20):
    N = trajectory.shape[0]
    def dist_to_all(point):
        return jnp.linalg.norm(trajectory - point, axis=-1)

    distance_matrix = jax.vmap(dist_to_all)(trajectory)

    radii = jnp.logspace(jnp.log10(r_min), jnp.log10(r_max), num=num_r)

    def compute_c_r(r):
        closer_pairs = jnp.sum(distance_matrix < r) - N
        return closer_pairs / (N * (N - 1))

    C_r = jax.vmap(compute_c_r)(radii)

    safe_mask = (C_r > 1e-8) & (radii > 0)
    log_r = jnp.log(radii[safe_mask])
    log_C = jnp.log(C_r[safe_mask])

    log_r_mean = jnp.mean(log_r)
    log_C_mean = jnp.mean(log_C)

    numerator = jnp.sum((log_r - log_r_mean) * (log_C - log_C_mean))
    denominator = jnp.sum((log_r - log_r_mean) ** 2)

    fractal_dimension = numerator / (denominator + 1e-15)

    return fractal_dimension, radii, C_r

# finding maxima
def find_maxima_in_z(traj):
    z_trj_left = traj[:-2, 2]
    z_trj_center = traj[1:-1, 2]
    z_trj_right = traj[2:, 2]

    is_maximum = (z_trj_center > z_trj_left) & (z_trj_center > z_trj_right)

    maximal_values = z_trj_center[is_maximum]
    return maximal_values

def run_trajectory_uniqueness_test(model, base_ic, num_samples=50, perturbation_scale=1e-5, test_steps=2000, tol=1e-2):
    print(f"Running Trajectory Test ({num_samples} Perturbations)")

    noise = jrandom.normal(master_key, shape=(num_samples, 3)) * perturbation_scale
    perturbed_ics = base_ic + noise

    vectorized_rollout = jax.vmap(lambda ic: prediction_func(model, ic, test_steps))
    all_emu_trajs = vectorized_rollout(perturbed_ics)

    final_states = all_emu_trajs[:, -1, :]

    diffs = final_states[:, None, :] - final_states[None, :, :]
    dist_matrix = jnp.linalg.norm(diffs, axis=-1)

    rounded_distances = jnp.round(dist_matrix / tol) * tol
    unique_final_clusters = jnp.unique(rounded_distances, axis=0)
    num_unique_trajectories = len(unique_final_clusters)
    
    uniqueness_ratio = (num_unique_trajectories / num_samples) * 100
    
    print(f"Perturbation Scale: {perturbation_scale}")
    print(f"Tolerance: {tol}")
    print(f"Unique Trajectories: {num_unique_trajectories} / {num_samples} ({uniqueness_ratio:.1f}%)")
    
    if num_unique_trajectories == 1:
        print("TEST FAILED")
        print("-> Perturbed path collapsed")
    elif uniqueness_ratio < 50.0:
        print("TEST WARNING")
        print("-> Trajectories are grouping together heavily")
    else:
        print("TEST PASSED")
        print("-> Variations successfully exploded into unique paths.")
        
    return all_emu_trajs, num_unique_trajectories

