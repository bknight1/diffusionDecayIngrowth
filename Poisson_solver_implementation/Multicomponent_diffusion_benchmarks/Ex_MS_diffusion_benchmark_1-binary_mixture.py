# %% [markdown]
# ### Benchmark of coupled diffusion
#
# Outlined in [Chai et al., 2019](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.99.023312)
#
#
# Based on benchmarks in [Crank, 1975](https://www-eng.lbl.gov/~shuman/NEXT/MATERIALS&COMPONENTS/Xe_damage/Crank-The-Mathematics-of-Diffusion.pdf), p. 118
#
#
# Included in multicomponent diffusion papers:
# - [Perchuk & Gerya, 2005](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=973b9a24a4dac26a8631800752cc2dd8b0676551)
# - [Loomis et al., 1985](https://link.springer.com/content/pdf/10.1007/BF00373040.pdf)
# - 
#
#
# Garnet multicomponent diffusion papers (outlining the coupling)
# - [Gaides et al., 2008](https://doi.org/10.1007/s00410-007-0263-z)
# - [Perchuk & Gerya, 2005](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=973b9a24a4dac26a8631800752cc2dd8b0676551) (has a two-component implementation too)
# - [Faryad et al., 2022](https://doi.org/10.1093/petrology/egac118)
# - [Florence & Spear, 1991](https://link.springer.com/article/10.1007/BF00310683)
# - [Tirone & Ganguly, 2010](https://www.sciencedirect.com/science/article/pii/S1342937X1000002X)

# %% [markdown]
# #### Ficks first law:
#
# $$
# J = -D \frac{dC}{dx}
# $$
#
# Where:
# - _J_ is the diffusion flux, representing the amount of substance that will flow through a unit area per unit time (e.g., mol/m²·s).
# - _D_ is the diffusion coefficient (or diffusivity), a constant that depends on the material and conditions (e.g., temperature).
# - _C_ is the concentration of the diffusing substance.
# - $\frac{dC}{dx}$ is the concentration gradient along the $x$-axis (change in concentration with respect to position).
#
# Expanded for multicomponent diffusion:
#
# $$
# \mathbf{J}_i = -\sum_{j=1}^{N} D_{ij} \nabla C_j
# $$
#
# Where:
# - $\mathbf{J}_i$ is the diffusion flux of component _i_.
# - $D_{ij}$ is the diffusion coefficient matrix, where each $D_{ij}$ represents the diffusion coefficient of component _i_ with respect to the concentration gradient of component _j_.
# - $\nabla C_j$ is the concentration gradient of component $j$.
#

# %% [markdown]
# # Benchmark 1
# A simple two-component coupled diffusion problem
#
# - Diffusion of $\xi_1$
# - $\xi_2$ = 1 - $\xi_1$
#
#
#
# ### Initial and boundary conditions
# $$
# t = 0 : \xi_1 = C_0, \, x < 0, \, \xi_1 = C_1, \, x \geq 0,
# $$
# $$
# x = -\infty, \, \xi_1 = C_0, \, x = +\infty, \, \xi_1 = C_1.
# $$
#
# ### Analytical solution for $\xi_1$
# $$
# \xi_1 = \frac{C_0 + C_1}{2} + \frac{C_1 - C_0}{2} \, \text{erf} \left( \frac{x}{2 \sqrt{D t}} \right),
# $$
#
# $$
# \xi_2 = 1 - \xi_1
# $$
#
# ###  Analytical solution for diffusion flux $(J_1)$
# $$
# J_1 = - D \nabla \xi_1 = - \frac{C_1 - C_0}{2} \sqrt{\frac{D}{\pi t}} e^{-\frac{x^2}{4 D t}},
# $$
#

# %% [markdown]
# ### UW implementation

# %%
import underworld3 as uw
import numpy as np
import math

import sympy as sp

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt

# %%
csize = uw.options.getReal(name='csize', default = 0.01)

degree = uw.options.getInt(name='degree', default = 1)


kappa = uw.options.getReal(name='kappa', default = 0.05)

C0 = uw.options.getReal(name='C0_val', default = 0.9)
C1 = uw.options.getReal(name='C1_val', default = 0.1)

# ### Temp to remain constant for model run
# Temp = uw.options.getReal(name='temp', default = 800) ### C


# model_duration = uw.options.getReal(name='duration', default = 500) ### Myr 


# %%
import os

outputPath = './output/multicomponent_diffusion_benchmarks/'

os.makedirs(outputPath, exist_ok=True)

# %%
mesh_qdegree = degree
mesh_qdegree

# %%
mesh = uw.meshing.StructuredQuadBox( minCoords=(-6, -1), maxCoords=(6,1), elementRes=(96,96) )





# %%
A = uw.discretisation.MeshVariable("A", mesh, 1, degree=degree, varsymbol=r'C_A')
A_star = uw.discretisation.MeshVariable("A_star", mesh, 1, degree=degree, varsymbol=r'C_{A^*}')

B = uw.discretisation.MeshVariable("B", mesh, 1, degree=degree, varsymbol=r'C_B')



with mesh.access(A, A_star, B):
    A.data[:,0] = C1
    
    A.data[:,0][A.coords[:,0] < 0.] = C0

    A_star.data[...] = A.data[...]

    B.data[:,0] = 1 - A.data[:,0]

    # B.data[:,0] = C_min
    # B.data[:,0][B.coords[:,0] > 0.5] = C_max

    # B_star.data[...] = B.data[...]

    

# %%
with mesh.access(A):
    cbar = plt.scatter(A.coords[:,0], A.coords[:,1], c=A.data[:,0])
    plt.colorbar(cbar)


# %% [markdown]
# ###  Setting up the diffusion flux $(J_1)$ term
# $$
# J_1 = - D \nabla \xi_1 = - \frac{C_1 - C_0}{2} \sqrt{\frac{D}{\pi t}} e^{-\frac{x^2}{4 D t}},
# $$

# %%
def gradient_calc(C_sym):
    """Calculates the gradient of concentration C in the x and y component"""
    # Assume x, y as spatial coordinates
    if mesh.dim == 3:
        x, y, z = mesh.X
    else:
        x, y = mesh.X

    # Compute the gradients of each concentration
    grad_C_x = sp.Matrix([sp.diff(C_sym, x)])
    grad_C_y = sp.Matrix([sp.diff(C_sym, y)])

    gradients = sp.Matrix([grad_C_x, grad_C_y])
    
    if mesh.dim == 3:
        grad_C_z = sp.Matrix([sp.diff(C_sym, z)])

        gradients = sp.Matrix([grad_C_x, grad_C_y, grad_C_z])

    return gradients


# %%
from sympy import symbols, Symbol, S, prod, Matrix
def create_B(n):
    # Create symbols for xi_1 to xi_{n-1}
    xi = {}
    for i in range(1, n):
        xi[i] = Symbol(f'xi_{i}')
    
    # Create symbols for D_{ij}, where i ≠ j and i ∈ [1, n-1], j ∈ [1, n]
    D = {}
    for i in range(1, n):
        for j in range(1, n + 1):
            if i != j:
                D[(i, j)] = Symbol(f'D_{i}{j}')
    
    # Initialize the (n-1) x (n-1) matrix B with zeros
    B = Matrix(n - 1, n - 1, lambda i, j: S.Zero)
    
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                # Off-diagonal entries
                B[i - 1, j - 1] = xi[i] * (1 / D[(i, n)] - 1 / D[(i, j)])
            else:
                # Diagonal entries
                term1 = 1 / D[(i, n)]
                sum_terms = S.Zero
                for k in range(1, n):
                    if k != i:
                        sum_terms += xi[k] * (1 / D[(i, k)] - 1 / D[(i, n)])
                B[i - 1, i - 1] = term1 + sum_terms
    
    return B


# %%
B_term = create_B(2)

# %%
D_12 = sp.Symbol('D_12')

# %%
D_tilda = B_term.inv()[0]

# %%
D_tilda

# %%
D_tilda.subs({D_12:kappa})

# %%
A_flux_vector = D_tilda.subs({D_12:kappa}) * gradient_calc(A.sym)
A_flux_vector

# %%
A_flux_vector_star = A_flux_vector.copy()

# %%
max_dt = mesh.get_min_radius()**2/kappa
dt = max_dt

# %% [markdown]
# #### Using Poisson solver

# %%
Diffusion_A = uw.systems.Poisson(mesh, A)


# %%
Diffusion_A.petsc_options["snes_monitor_short"] = None
Diffusion_A.petsc_options["snes_converged_reason"] = None


# %%
def solve_diff_eq(solver, u, u_star, flux_vector, flux_vector_star, dt):
    ### setup f0 term
    solver.f = - ((u.sym[0] - u_star.sym[0]) / dt)

    
    

    ### setup f1 term
    theta = 0
    diffusion_CM = uw.constitutive_models.DiffusionModel
    diffusion_CM.flux = flux_vector

    solver.constitutive_model =  diffusion_CM

    solver.flux = theta*flux_vector + ((1-theta)*flux_vector_star)

    ### solve
    solver.solve()

    ### update history terms
    # flux_vector_star = flux_vector.copy()
    
    with mesh.access(u, u_star):
        u_star.data[...] = u.data[...]

# %%
total_time = 0.

# %%
model_duration = 1.

while total_time < model_duration:
    if total_time + dt > model_duration:
        dt = model_duration - total_time

    with mesh.access(A, B):
        B.data[:,0] = 1 - A.data[:,0]
        
    solve_diff_eq(Diffusion_A, A, A_star, A_flux_vector, A_flux_vector_star, dt)

    
    A_flux_vector_star = A_flux_vector.copy()
    total_time += dt


# %%
with mesh.access(A):
    cbar = plt.scatter(A.coords[:,0], A.coords[:,1], c=A.data[:,0], s=1, marker='s')
    plt.colorbar(cbar)

# %% [markdown]
# ### Analytical solution for $\xi_1$
# $$
# \xi_1 = \frac{C_0 + C_1}{2} + \frac{C_1 - C_0}{2} \, \text{erf} \left( \frac{x}{2 \sqrt{D t}} \right),
# $$
#

# %%
# Define the variables

# x, t, D = sp.symbols('x t D')
# C0_sym, C1_sym = sp.symbols('C0 C1')

# # Define the error function erf
# erf = sp.erf(x / (2 * sp.sqrt(D * t)))

# # Define the equation for ξ1 (68a)
# xi_1 = (C0 + C1) / 2 + (C1 - C0) / 2 * erf


from scipy.special import erf as scipy_erf

# Update xi_1_func to handle numpy arrays using scipy's error function
def xi_1_numpy(x_array, t, D, C0, C1):
    return ((C0 + C1) / 2) + (((C1 - C0) / 2) * scipy_erf(x_array / (2 * np.sqrt(D * t))))



# %%
x_arr = np.arange(-6+0.01, 6, 0.01)
y_arr = np.zeros_like(x_arr)
coords = np.column_stack([x_arr, y_arr])

# %%
data_t1 = uw.function.evaluate(A.sym, coords)
plt.plot(x_arr, data_t1, label='UW')
plt.plot(x_arr, xi_1_numpy(x_arr, model_duration, kappa, C0, C1), label='analytical', ls=':' )
plt.legend()

# %%
model_duration = 5.

while total_time < model_duration:
    if total_time + dt > model_duration:
        dt = model_duration - total_time
        
    solve_diff_eq(Diffusion_A, A, A_star, A_flux_vector, A_flux_vector_star, dt)
    A_flux_vector_star = A_flux_vector.copy()
    total_time += dt


# %%
data_t5 = uw.function.evaluate(A.sym, coords)

# %%

plt.plot(x_arr, xi_1_numpy(x_arr, 0., kappa, C0, C1), label='t=0', ls='--', c='k', alpha=0.5 )


plt.plot(x_arr, data_t1, label='UW (t=1)', c='red')
plt.plot(x_arr, xi_1_numpy(x_arr, 1., kappa, C0, C1), label='analytical (t=1)', ls='-.', c='k', alpha=0.5 )

plt.plot(x_arr, data_t5, label='UW (t=5)', c='green')
plt.plot(x_arr, xi_1_numpy(x_arr, 5., kappa, C0, C1), label='analytical (t=5)', ls=':', c='k', alpha=0.5 )

plt.legend()

plt.ylabel(r'$\xi_1$')
plt.xlabel(r'$x$')

plt.savefig('benchmark1.pdf')
