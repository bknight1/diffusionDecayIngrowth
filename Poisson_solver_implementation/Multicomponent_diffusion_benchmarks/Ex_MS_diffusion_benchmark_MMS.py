# %% [markdown]
# ## Benchmark of Maxwell-Stefan diffusion using MMS
# - [McLeod and Bourgault, 2014](https://www.sciencedirect.com/science/article/pii/S0045782514002321#f000010)

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
# ## 

# %% [markdown]
# # Benchmark using MMS
# - Using method of manufactured solutions to determine the concentration field.
# - This is a Poisson/time independent benchmark.
#
# ### Initial and boundary conditions
#
# - Domain size of _0,0_ to _1,1_
#
# two functions $ f_1 $ and $ f_2 $ as follows:
#
# $$
# f_1 =
# \begin{cases}
# \frac{\sinh\left(\frac{\pi}{2}\right) \sin\left(\frac{\pi x}{2}\right)}{\pi^2}, & x \in [0, 1], \quad y = 1, \\
# \frac{\sinh\left(\frac{\pi y}{2}\right)}{\pi^2}, & x = 1, \quad y \in [0, 1], \\
# 0, & \text{Otherwise}.
# \end{cases}
# $$
#
# $$
# f_2 =
# \begin{cases}
# \frac{\cosh\left(\frac{\pi}{2}\right) \cos\left(\frac{\pi x}{2}\right)}{\pi^2}, & x \in [0, 1], \quad y = 1, \\
# \frac{\cos\left(\frac{\pi x}{2}\right)}{\pi^2}, & x \in [0, 1], \quad y = 0, \\
# \frac{\cosh\left(\frac{\pi y}{2}\right)}{\pi^2}, & x = 0, \quad y \in [0, 1], \\
# 0, & \text{Otherwise}.
# \end{cases}
# $$
#
# For this test case, Dirichlet boundary conditions are used on all $ \Gamma $, i.e., $ \Gamma_D = \Gamma $. The reaction rates are defined by two functions $ r_1 $ and $ r_2 $ in the domain $ \Omega $ as follows:
#
# $$
# r_1 = 
# \left( \frac{\alpha_{21}}{\chi} - \frac{\alpha_{21} \beta_2}{D_{12} \xi^2} + \frac{\alpha_{12} \tilde{\alpha}_2}{D_{23} \xi^2} \right) 
# \frac{\left( \sin(\pi x / 2) + \sinh(\pi y / 2) \right)}{4\pi^2},
# $$
#
# $$
# r_2 = 
# \left( \frac{\alpha_{12}}{\chi} - \frac{\alpha_{12} \tilde{\alpha}_1}{D_{23} \xi^2} + \frac{\beta_1 \alpha_{21}}{D_{13} \xi^2} \right) 
# \frac{\left( \sin(\pi x / 2) + \sinh(\pi y / 2) \right)}{4\pi^2}.
# $$
#
# where the term $ \chi $ is defined as:
#
# $$
# \chi = \frac{1}{D_{13} D_{23}} - \frac{\alpha_{12} \xi_1}{D_{23}} - \frac{\alpha_{21} \xi_2}{D_{13}}.
# $$
#
# The exact solution of the Maxwell–Stefan equations for these data is given by:
#
# $$
# \xi_1 = \frac{\sin(\pi x / 2) \sinh(\pi y / 2)}{\pi^2}, \quad
# \xi_2 = \frac{\cos(\pi x / 2) \cosh(\pi y / 2)}{\pi^2},
# $$
#
# $$
# J_1 = -\frac{\beta_2}{\chi} \nabla \xi_1 + \frac{\tilde{\alpha}_2}{\chi} \nabla \xi_2,
# \quad
# J_2 = \frac{\beta_1}{\chi} \nabla \xi_1 - \frac{\tilde{\alpha}_1}{\chi} \nabla \xi_2.
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


D12 = uw.options.getReal(name='D_12', default = 5)
D13 = uw.options.getReal(name='D_13', default = 10)
D23 = uw.options.getReal(name='D_13', default = 5)


model_duration = uw.options.getReal(name='duration', default =1.)


# %%
import os

outputPath = './output/multicomponent_diffusion_benchmarks/'

os.makedirs(outputPath, exist_ok=True)

# %%
mesh_qdegree = degree
mesh_qdegree

# %%

mesh = uw.meshing.StructuredQuadBox( minCoords=(0, 0), maxCoords=(1, 1), elementRes=(50,50) )



# %%
A = uw.discretisation.MeshVariable("A", mesh, 1, degree=degree, varsymbol=r'C_A')
A_star = uw.discretisation.MeshVariable("A_star", mesh, 1, degree=degree, varsymbol=r'C_{A^*}')

B = uw.discretisation.MeshVariable("B", mesh, 1, degree=degree, varsymbol=r'C_B')
B_star = uw.discretisation.MeshVariable("B_star", mesh, 1, degree=degree, varsymbol=r'C_{B^*}')

C = uw.discretisation.MeshVariable("C", mesh, 1, degree=degree, varsymbol=r'C_C')


# %% [markdown]
# ### Setup solvers

# %%
A_diffusion = uw.systems.Poisson(mesh, u_Field=A)
B_diffusion = uw.systems.Poisson(mesh, u_Field=B)

# %%
for _solver in [A_diffusion, B_diffusion]:
    _solver.petsc_options['snes_rtol'] = 1e-12
    _solver.petsc_options['snes_atol'] = 1e-6
    
    ### see the SNES output
    _solver.petsc_options["snes_converged_reason"] = None

# %%
import sympy as sp

# Step 1: Define symbolic variables
x, y = mesh.X

# Step 2: Define the mathematical expressions for f1 and f2
f1_expr_1 = sp.sinh(sp.pi / 2) * sp.sin(sp.pi * x / 2) / sp.pi**2
f1_expr_2 = sp.sinh(sp.pi * y / 2) / sp.pi**2

f2_expr_1 = sp.cosh(sp.pi / 2) * sp.cos(sp.pi * x / 2) / sp.pi**2
f2_expr_2 = sp.cos(sp.pi * x / 2) / sp.pi**2
f2_expr_3 = sp.cosh(sp.pi * y / 2) / sp.pi**2

# Step 3: Define f1 as a piecewise function
f1 = sp.Piecewise(
    (f1_expr_1, sp.Eq(y, 1)),  # y = 1
    (f1_expr_2, sp.Eq(x, 1)),  # x = 1
    (0, True)  # Otherwise
)

# Step 4: Define f2 as a piecewise function 
f2 = sp.Piecewise(
    (f2_expr_1, sp.Eq(y, 1)),  # y = 1
    (f2_expr_2, sp.Eq(y, 0)),  # y = 0
    (f2_expr_3, sp.Eq(x, 0)),  # x = 0
    (0, True)  # Otherwise
)




# %% [markdown]
# ### boundary conditions
# $\xi_{i}$ = f$_i$

# %%
for boundary in mesh.boundaries:
    boundary_label = boundary.name
    A_diffusion.add_dirichlet_bc([f1], boundary_label)
    B_diffusion.add_dirichlet_bc([f2], boundary_label)

# %%
A_diffusion.constitutive_model = uw.constitutive_models.DiffusionModel
A_diffusion.f = f1
A_diffusion.solve()

# %%
with mesh.access(A):
    plt.scatter(A.coords[:,0], A.coords[:,1], c=A.data[:,0])

# %%
B_diffusion.constitutive_model = uw.constitutive_models.DiffusionModel
B_diffusion.f = f2
B_diffusion.solve()

# %% [markdown]
# ### Check results against analytical solutions

# %%
A_analytical = (sp.sin((sp.pi*x)/2)*sp.sinh((sp.pi*y)/2))/sp.pi**2
A_L2 = sp.sqrt( (A.sym[0] - A_analytical)**2 ) 

# %%
with mesh.access(A):
    c = plt.scatter(A.coords[:,0], A.coords[:,1], c=uw.function.evaluate(A_L2, A.coords) )
    plt.colorbar(c)

# %%
B_analytical = (sp.cos((sp.pi*x)/2)*sp.cosh((sp.pi*y)/2))/sp.pi**2
B_L2 = sp.sqrt( (B.sym[0] - B_analytical)**2 ) 

# %%
with mesh.access(B):
    cbar = plt.scatter(B.coords[:,0], B.coords[:,1], c=uw.function.evaluate(B_L2, A.coords))
    plt.colorbar(cbar)
