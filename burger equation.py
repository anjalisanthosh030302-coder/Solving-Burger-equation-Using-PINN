import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1. Neural Network
# ---------------------------
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 1)
        )

    def forward(self, x, t):
        input = torch.cat([x, t], dim=1)
        return torch.tanh(self.net(input))   

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

model = PINN().to(device)
model.apply(init_weights)


# ---------------------------
# 2. Training Data
# ---------------------------

# Boundary points
N_bc = 25
t_bc = torch.linspace(0, 1, N_bc).view(-1, 1)
x_bc1 = -1 * torch.ones_like(t_bc)
x_bc2 = 1 * torch.ones_like(t_bc)
u_bc1 = torch.zeros_like(t_bc)
u_bc2 = torch.zeros_like(t_bc)

# Initial condition
N_ic = 50
x_ic = torch.linspace(-1, 1, N_ic).view(-1, 1)
t_ic = torch.zeros_like(x_ic)
u_ic = -torch.sin(np.pi * x_ic)

# Collocation points
N_f = 4000
x_f = torch.cat([
    torch.rand(int(0.7*N_f), 1)*2 - 1,
    torch.randn(int(0.3*N_f), 1)*0.2   # focus near 0
])
x_f = torch.clamp(x_f, -1, 1)
t_f = torch.rand(N_f, 1)

# Move to device
x_f, t_f = x_f.to(device), t_f.to(device)
x_ic, t_ic, u_ic = x_ic.to(device), t_ic.to(device), u_ic.to(device)
x_bc1, t_bc, u_bc1 = x_bc1.to(device), t_bc.to(device), u_bc1.to(device)
x_bc2, _, u_bc2 = x_bc2.to(device), t_bc.to(device), u_bc2.to(device)


def normalize(x):
    xmin = x.min()
    xmax = x.max()
    
    if (xmax - xmin) == 0:
        return x   # ⚠️ avoid division by zero
    
    return 2*(x - xmin)/(xmax - xmin) - 1


t_f = normalize(t_f)
t_ic = normalize(t_ic)
t_bc = normalize(t_bc)

# ---------------------------
# 3. Physics Loss
# ---------------------------
def pde_loss(model, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    nu = 0.01 / np.pi
    f = u_t + u * u_x - nu * u_xx

    return torch.mean(f**2)

# ---------------------------
# 4. Boundary + Initial Loss
# ---------------------------
def data_loss(model):
    u_ic_pred = model(x_ic, t_ic)
    u_bc1_pred = model(x_bc1, t_bc)
    u_bc2_pred = model(x_bc2, t_bc)

    loss_ic = torch.mean((u_ic_pred - u_ic) ** 2)
    loss_bc1 = torch.mean((u_bc1_pred - u_bc1) ** 2)
    loss_bc2 = torch.mean((u_bc2_pred - u_bc2) ** 2)

    return loss_ic + loss_bc1 + loss_bc2

# ---------------------------
# 5. Training
# ---------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
for epoch in range(12000):
    optimizer.zero_grad()

    loss_f = pde_loss(model, x_f, t_f)
    loss_u = data_loss(model)

    
    pde_weight = min(1.0, epoch / 3000)
    
    loss = pde_weight * loss_f + 10 * loss_u
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()

    if epoch % 500 == 0:
       print(f"Epoch {epoch}, Loss: {loss.item()}")
       
       
model.train()        
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=500,
    tolerance_grad=1e-9,
    tolerance_change=1e-9,
    history_size=50
)
def closure():
    optimizer_lbfgs.zero_grad()
    
    loss_f = pde_loss(model, x_f, t_f)
    loss_u = data_loss(model)
    
    loss = 1.0 * loss_f + 10 * loss_u  
    
    loss.backward()
    return loss

print("Starting L-BFGS...")
for i in range(5):
    optimizer_lbfgs.step(closure)

def solve_burgers(X, t, nu):
    X = X.numpy().flatten() 
    U = np.zeros_like(X)

    def f(y):
        return np.exp(-np.cos(np.pi*y)/(2*np.pi*nu))

    def g(y):
        return np.exp(-(y**2)/(4*nu*t))

    eta_vals = np.linspace(-5, 5, 200)

    for i in range(len(X)):
        x = X[i]

        if abs(x) != 1:
            def integrand1(eta):
                return np.sin(np.pi*(x-eta)) * f(x-eta) * g(eta)

            def integrand2(eta):
                return f(x-eta) * g(eta)

            num_vals = np.array([integrand1(e) for e in eta_vals])
            den_vals = np.array([integrand2(e) for e in eta_vals])

            num = np.trapezoid(num_vals, eta_vals)
            den = np.trapezoid(den_vals, eta_vals)

            U[i] = -num / den

    return U
# ---------------------------
# 6. Prediction + Comparison
# ---------------------------

x_test = torch.linspace(-1, 1, 1000).view(-1,1).to(device)
t_test = torch.ones_like(x_test) * 0.5

# PINN prediction
u_pred = model(x_test, t_test).detach().cpu().numpy()

# Exact solution
nu = 0.01 / np.pi
u_exact = solve_burgers(x_test.cpu(), 0.5, nu)

# Plot
plt.figure(figsize=(8,5))
plt.plot(x_test.cpu(), u_pred, label="PINN", linewidth=2)
plt.plot(x_test.cpu(), u_exact, '--', label="Exact", linewidth=2)

plt.title("PINN vs Exact Solution (t=0.5)")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.grid()
plt.show()

error = np.linalg.norm(u_pred.flatten() - u_exact) / np.linalg.norm(u_exact)
print("Relative L2 Error:", error)


