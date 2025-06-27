# This script demonstrates how to download ERA5 temperature data, preprocess it, and train a Physics-Informed Neural Network (PINN) to model the 2D heat equation in Europe.
# Structured as a Jupyter notebook with code cells.

# %% Download 5 days of 6-hourly temperature data from ERA5 (Europe)
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': '2m_temperature',
        'year': '2022',
        'month': '07',
        'day': ['01', '02', '03', '04', '05'],
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'format': 'netcdf',
        'area': [60, -10, 35, 30],  # [N, W, S, E] -> Europe-ish
    },
    'era5_europe_temp.nc'
)



# %% Inspect and plot the data using xarray and matplotlib

import xarray as xr
import matplotlib.pyplot as plt

# Load NetCDF file
ds = xr.open_dataset("era5_europe_temp.nc")
print(ds)
ds = ds.rename({'valid_time': 'time'})

# Extract temperature variable
temp = ds['t2m'] - 273.15  # Convert from Kelvin to Celsius

# Plot first timestep
temp.isel(time=0).plot(cmap='coolwarm')
plt.title("Temperature at t=0")
plt.show()

# %% Extract data as input and output 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load ERA5 temperature data and convert to Celsius
ds = xr.open_dataset("era5_europe_temp.nc")
ds = ds.rename({'valid_time': 'time'})  # Rename 

temp = ds['t2m'] - 273.15  # Kelvin to Celsius

# Flip latitude if decreasing (ERA5 often is)
if np.all(np.diff(temp.latitude.values) < 0):
    temp = temp.sortby('latitude')

# Extract coordinates
lats = temp.latitude.values  # increasing order
lons = temp.longitude.values

# Get temperature array (time, lat, lon)
T = temp.values
nt, ny, nx = T.shape

times = temp.time.values
time_days = (times - times[0]) / np.timedelta64(1, 'D')  # e.g. [0.0, 0.25, 0.5, ...]

# Make meshgrid
lon_grid, lat_grid, time_grid = np.meshgrid(lons, lats, time_days, indexing='xy')


# Flatten to create inputs: (lon, lat, time)
inputs = np.stack([
    lon_grid.ravel(),   # longitude
    lat_grid.ravel(),   # latitude
    time_grid.ravel()   # days since start
], axis=1)

# Flatten temperature outputs
outputs = T.reshape(-1, 1)


# %% Definitions of helper functions and classes for PINN
# We will use a simple feedforward neural network to approximate the temperature field.

# We want to model the 2D heat equation:
# ∂T/∂t = α * (∂²T/∂x² + ∂²T/∂y²)
# where α is the thermal diffusivity constant (non-learnable here)

import torch
import torch.nn as nn

# Simple feedforward fully connected network
class FCNN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, out_dim=1, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PINNLoss(nn.Module):
    def __init__(self, model, lambda_pde=0.01, log_alpha=True):
        super().__init__()
        self.model = model
        self.log_lambda = nn.Parameter(torch.tensor([-2.0]))  # ln(0.135), for example


        # Learnable alpha: log(alpha) to ensure positivity
        if log_alpha:
            self.log_alpha = nn.Parameter(torch.tensor([-4.0]))  # ln(1e-2)
        else:
            self.alpha = nn.Parameter(torch.tensor([1e-2]))
            self.log_alpha = None

    def get_alpha(self):
        if self.log_alpha is not None:
            return torch.exp(self.log_alpha)
        else:
            return self.alpha

    def forward(self, x_data, y_data, x_phys):
        alpha = self.get_alpha()
        lambda_pde = torch.exp(self.log_lambda)


        # ==== Data loss ====
        y_pred = self.model(x_data)
        data_loss = torch.mean((y_pred - y_data) ** 2)

        # ==== PDE residual ====
        x_phys.requires_grad_(True)
        T = self.model(x_phys)

        grads = torch.autograd.grad(T, x_phys, grad_outputs=torch.ones_like(T),
                                    create_graph=True)[0]
        dT_dx = grads[:, 0:1]
        dT_dy = grads[:, 1:2]
        dT_dt = grads[:, 2:3]

        d2T_dx2 = torch.autograd.grad(dT_dx, x_phys, grad_outputs=torch.ones_like(dT_dx),
                                      create_graph=True)[0][:, 0:1]
        d2T_dy2 = torch.autograd.grad(dT_dy, x_phys, grad_outputs=torch.ones_like(dT_dy),
                                      create_graph=True)[0][:, 1:2]

        residual = dT_dt - alpha * (d2T_dx2 + d2T_dy2)
        pde_loss = torch.mean(residual ** 2)

        total_loss = data_loss + lambda_pde * pde_loss

        return total_loss, data_loss, pde_loss, alpha.item(), lambda_pde.item()
    

def sample_X_phys(n_points, lon_bounds, lat_bounds, time_bounds):
    lon_min, lon_max = lon_bounds
    lat_min, lat_max = lat_bounds
    time_min, time_max = time_bounds

    lon = np.random.uniform(lon_min, lon_max, n_points)
    lat = np.random.uniform(lat_min, lat_max, n_points)
    time = np.random.uniform(time_min, time_max, n_points)

    X_phys = np.stack([lon, lat, time], axis=1)
    return torch.tensor(X_phys, dtype=torch.float32)

def compute_residuals(model, xyt, alpha):
    xyt.requires_grad_(True)
    T_pred = model(xyt)

    grads = torch.autograd.grad(
        T_pred, xyt, 
        grad_outputs=torch.ones_like(T_pred),
        create_graph=True
    )[0]

    dT_dt = grads[:, 2:3]
    dT_dx = torch.autograd.grad(grads[:, 0:1], xyt, grad_outputs=torch.ones_like(dT_dt), create_graph=True)[0][:, 0:1]
    dT_dy = torch.autograd.grad(grads[:, 1:2], xyt, grad_outputs=torch.ones_like(dT_dt), create_graph=True)[0][:, 1:2]

    d2T_dx2 = torch.autograd.grad(dT_dx, xyt, grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0][:, 0:1]
    d2T_dy2 = torch.autograd.grad(dT_dy, xyt, grad_outputs=torch.ones_like(dT_dy), create_graph=True)[0][:, 1:2]

    residual = dT_dt - alpha * (d2T_dx2 + d2T_dy2)
    return residual.detach().cpu().numpy()



def simple_data_loss(model, xyt, T_true):
    T_pred = model(xyt)
    return torch.mean((T_pred - T_true) ** 2)






# %% Import necessary libraries
import numpy as np
import torch
import torch.nn as nn


# %%
# Convert to torch tensors
X = torch.tensor(inputs, dtype=torch.float32)
Y = torch.tensor(outputs, dtype=torch.float32)

X_sub = X
Y_sub = Y
# %%
# Optional: subsample if needed (for speed)
idx = torch.randperm(X.shape[0])[:200000]
X_sub = X[idx]
Y_sub = Y[idx]
# %%
# Prepare physical space for PINN
    


# Use same ranges as in data
lon_min, lon_max = lons.min(), lons.max()
lat_min, lat_max = lats.min(), lats.max()
time_min, time_max = time_days.min(), time_days.max()
lon_bounds = (lons.min(), lons.max())
lat_bounds = (lats.min(), lats.max())
time_bounds = (time_days.min(), time_days.max())


# Create a grid: e.g. 40 x 40 x 10 points
n_lon, n_lat, n_time = 40, 40, 10
lon_lin = np.linspace(lon_min, lon_max, n_lon)
lat_lin = np.linspace(lat_min, lat_max, n_lat)
time_lin = np.linspace(time_min, time_max, n_time)

# Mesh and stack
lon_grid, lat_grid, time_grid = np.meshgrid(lon_lin, lat_lin, time_lin, indexing='xy')

X_phys = np.stack([
    lon_grid.ravel(),
    lat_grid.ravel(),
    time_grid.ravel()
], axis=1)

X_phys_tensor = torch.tensor(X_phys, dtype=torch.float32)


# %%

# === Training setup ===
model = FCNN(hidden_dim=128, depth=5)
loss_fn = PINNLoss(model)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3
)

# === Resume or Start Fresh ===
start_epoch = 0
resume = False
checkpoint_path = "checkpoint.pt"

if resume:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from epoch {start_epoch}")

# === Training loop ===
n_epochs = 4000
n_phys = 200000

for epoch in range(start_epoch, start_epoch + n_epochs):
    X_phys_tensor = sample_X_phys(n_phys, lon_bounds, lat_bounds, time_bounds)

    optimizer.zero_grad()
    loss, data_l, pde_l, alpha_val, lambda_val = loss_fn(X_sub, Y_sub, X_phys_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:>4} | Total: {loss.item():.5f} | Data: {data_l:.5f} | PDE: {pde_l:.5f} | α: {alpha_val:.2f} | λ: {lambda_val:.2f}")

    if epoch % 1000 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_fn_state_dict": loss_fn.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")




# %% Save the model if you want to use it later
torch.save(model.state_dict(), "pinn_model_continuel8r.pt")


# %% Evaluation of PINN predictions
import torch
import torch.nn as nn

# load the trained model
model = FCNN()
model.load_state_dict(torch.load("pinn_model.pt"))
model.eval()


# %% # Evaluate the model on a specific day

# Choose day index (e.g., day 3)
t_eval = 3  # in days (since we used time_days)

# Create lat/lon meshgrid (same as before)
lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='xy')

# Flatten and combine with constant time
lon_flat = lon_grid.ravel()
lat_flat = lat_grid.ravel()
t_flat = np.full_like(lon_flat, t_eval)

# Stack into input tensor
X_eval = np.stack([lon_flat, lat_flat, t_flat], axis=1)
X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

model.eval()

with torch.no_grad():
    T_pred = model(X_eval_tensor).cpu().numpy().reshape((ny, nx))

    # Find closest index to day 3
i = np.argmin(np.abs(time_days - t_eval))
T_true = T[i, :, :]  # shape: (lat, lon)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
cs1 = axs[0].contourf(lons, lats, T_true, levels=50, cmap='coolwarm')
axs[0].set_title(f"ERA5 True Temperature (Day {t_eval})")
fig.colorbar(cs1, ax=axs[0])

cs2 = axs[1].contourf(lons, lats, T_pred, levels=50, cmap='coolwarm')
axs[1].set_title(f"PINN Predicted Temperature (Day {t_eval})")
fig.colorbar(cs2, ax=axs[1])

plt.tight_layout()
plt.show()


# %% # Compute and visualize the PDE residuals


residuals = compute_residuals(model, X_eval_tensor, alpha=1e-2)
residuals_grid = residuals.reshape((ny, nx))

plt.figure(figsize=(7, 5))
cs = plt.contourf(lons, lats, np.abs(residuals_grid), levels=50, cmap='viridis')
plt.colorbar(cs, label='|PDE Residual|')
plt.title(f"PDE Residual Magnitude (Day {t_eval})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()




# %%
