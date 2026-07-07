def neural_prediction_func(model, initial_state, prediction_steps, dt):
    """
    Integrates dx/dt = model(state) step-by-step over the training horizon.
    """
    def neural_vf(t, state, args):
        return model(state)

    term = ODETerm(neural_vf)
    solver = Heun()  # Stable 2nd order RK method for training gradients

    def derivative_stepper(carry, _):
        solution = diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=dt,
            dt0=dt,
            y0=carry,
            saveat=SaveAt(t1=True)
        )
        next_state = solution.ys[-1]
        return next_state, next_state

    _, trajectory = jax.lax.scan(derivative_stepper, initial_state, None, length=prediction_steps)
    return trajectory

@eqx.filter_value_and_grad
def compute_loss(model, batch_init_states, batch_true_trajectories, prediction_steps, dt):
    """
    Computes trajectory MSE across the batch.
    """
    vmapped_rollout = jax.vmap(
        lambda ic: neural_prediction_func(model, ic, prediction_steps, dt)
    )
    predicted_trajectories = vmapped_rollout(batch_init_states)
    return jnp.mean((predicted_trajectories - batch_true_trajectories) ** 2)

@eqx.filter_jit
def model_stepper_der(model, opt_state, optimizer, batch_init_states, batch_true_trajectories, prediction_steps, dt):
    """
    Replaces your previous step compiler.
    """
    loss, grads = compute_loss(model, batch_init_states, batch_true_trajectories, prediction_steps, dt)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

## set up modified data for derivative model
# creating the trsting and training data set
init_states = jrandom.uniform(master_key, shape=(9, 3), minval=-15.0, maxval=15.0)
init_states = init_states.at[:, 2].add(25.0)
training_testing_trajs = jax.vmap(generate_singel_training_traj)(init_states)
# setting up training data
training_trajs = training_testing_trajs[:6]
testing_trajs = training_testing_trajs[6:]
testing_initial_states = testing_trajs[:, 0, :]

# Define the prediction horizon for the emulator
# This means the emulator will be trained to predict PRED_HORIZON steps ahead.
PRED_HORIZON = 50 # You can adjust this value as needed

all_x_data = []
all_y_data = []

# Generate training data for each trajectory using the training_dataset function
for traj in training_trajs:
    x_d, y_d = training_dataset(traj, horizon=PRED_HORIZON)
    all_x_data.append(x_d)
    all_y_data.append(y_d)

x_train_pre_noise = jnp.concatenate(all_x_data, axis=0)
y_train = jnp.concatenate(all_y_data, axis=0)

# Verify shapes after concatenation
print(f"x_train shape: {x_train_pre_noise.shape}")
print(f"y_train shape: {y_train.shape}")

# input noise
master_key, noise_key = jrandom.split(master_key)
noise_scale = 0.1

input_noise = jrandom.normal(noise_key, shape=x_train_pre_noise.shape) * noise_scale
x_train = x_train_pre_noise + input_noise

print(f"x_train shape (Noisy): {x_train.shape}")
print(f"y_train shape (Clean): {y_train.shape}")

# train the drivative model
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3)
)
model_derivative = new_63_model
opt_state = optimizer.init(eqx.filter(model_derivative, eqx.is_array))

epochs = 50
batch_size = 256
num_samples = x_train.shape[0]

# PRED_HORIZON = y_train.shape[1] # This is the maximum horizon, but we use active_horizon below
dt_step = 0.01

for epoch in range(epochs):
    master_key, subkey = jrandom.split(master_key)
    permutation = jrandom.permutation(subkey, num_samples)

    epoch_losses = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        if epoch < 10:
            active_horizon = 4
        elif epoch < 25:
            active_horizon = 12
        else:
            active_horizon = 24

        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, num_samples)

        batch_indices = permutation[batch_start:batch_end]

        x_batch_current = x_train[batch_indices]  # Shape: (Batch, 3)
        y_batch_current = y_train[batch_indices, :active_horizon, :]  # Shape: (Batch, active_horizon, 3)

        if len(x_batch_current) > 0:
            # Pass active_horizon and dt down to the ODE tracking wrapper
            model_derivative, opt_state, loss = model_stepper_der(
                model_derivative, opt_state, optimizer,
                x_batch_current, y_batch_current,
                active_horizon, dt_step
            )
            epoch_losses.append(loss)

    if epoch % 10 == 0 or epoch == epochs - 1:
        if epoch_losses:
            avg_loss = jnp.mean(jnp.array(epoch_losses))
            print(f"Epoch {epoch}, Last Batch Loss: {loss:.6f}, Average Epoch Loss: {avg_loss:.6f}")
        else:
            print(f"Epoch {epoch}, No batches processed.")
