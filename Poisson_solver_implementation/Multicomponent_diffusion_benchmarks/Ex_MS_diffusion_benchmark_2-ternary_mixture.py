# %% [markdown]
# ## Benchmark of coupled diffusion
#
#
# ### A three-component coupling diffusion problem
#
# #### Outlined in [Geiser, 2015](https://www.tandfonline.com/doi/full/10.1080/23311835.2015.1092913)
#
# Also in:
# - [Chai et al., 2019](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.99.023312)
# - [Boudin et al., 2012](https://hal.science/hal-00490511/)
# - Approximation of [Duncan and Toor, 1962](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.690080112)
#

# %% [raw]
# Garnet multicomponent diffusion papers which outlining the coupling:
# - [Gaides et al., 2008](https://doi.org/10.1007/s00410-007-0263-z)
# - [Perchuk & Gerya, 2005](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=973b9a24a4dac26a8631800752cc2dd8b0676551) (has a two-component implementation too)
# - [Faryad et al., 2022](https://doi.org/10.1093/petrology/egac118)
# - [Florence & Spear, 1991](https://link.springer.com/article/10.1007/BF00310683)
# - [Tirone & Ganguly, 2010](https://www.sciencedirect.com/science/article/pii/S1342937X1000002X)
#

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
# # Benchmark 2
# - Uphill diffusion of $\xi_1$ and $\xi_2$ in a 3 component system
#
# ### Initial and boundary conditions
#
# - Domain size of _0,0_ to _1,1_
# - Model duration of _1_
# $$
# \xi_1 = 
# \begin{cases} 
# 0.8, & 0 \leq x < 0.25, \\
# 1.6(0.75 - x), & 0.25 \leq x < 0.75, \\
# 0, & 0.75 \leq x \leq 1,
# \end{cases}
# $$
#
# $$
# \xi_2 = 0.2, \quad 0 \leq x \leq 1,
# $$
#
# where:
# $$
# \xi_3 = 1 - \xi_1 - \xi_2
# $$

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


D12 = uw.options.getReal(name='D_12', default = 0.833)
D13 = uw.options.getReal(name='D_13', default = 0.833)
D23 = uw.options.getReal(name='D_13', default = 0.168)


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

# %% [markdown]
# ### Set up the initial values
# $$
# \xi_1 = 
# \begin{cases} 
# 0.8, & 0 \leq x < 0.25, \\
# 1.6(0.75 - x), & 0.25 \leq x < 0.75, \\
# 0, & 0.75 \leq x \leq 1,
# \end{cases}
# $$
#
# $$
# \xi_2 = 0.2, \quad 0 \leq x \leq 1,
# $$

# %%
with mesh.access(A, A_star, B, B_star):
    A.data[:,0] = 0.8
    
    A.data[:,0][A.coords[:,0] >= 0.25] = 1.6*(0.75 - A.coords[:,0][A.coords[:,0] >= 0.25])
    
    A.data[:,0][A.coords[:,0] >= 0.75] = 0

    A_star.data[...] = A.data[...]

    B.data[:,0] = 0.2

    B_star.data[:,0] = B.data[:,0]


# %%
with mesh.access(A):
    cbar = plt.scatter(A.coords[:,0], A.coords[:,1], c=A.data[:,0])
    plt.colorbar(cbar)

# %% [markdown]
# #### Setting up the flux terms

# %% [markdown]
# ### System of Equations from references outlined above:
#
# $$
# \frac{1}{D_{13}} N_1 + \alpha N_1 \xi_2 - \alpha N_2 \xi_1 = -\nabla \xi_1,
# $$
#
# $$
# \frac{1}{D_{23}} N_2 - \beta N_1 \xi_2 + \beta N_2 \xi_1 = -\nabla \xi_2.
# $$
#
# ---
#
# ### Step 1: Rearrange the First Equation for $ N_1 $
#
# Rearrange the first equation:
#
# $$
# \left( \frac{1}{D_{13}} + \alpha \xi_2 \right) N_1 - \alpha \xi_1 N_2 = -\nabla \xi_1.
# $$
#
# So:
#
# $$
# N_1 = \left( \frac{1}{D_{13}} + \alpha \xi_2 \right)^{-1} \left( -\nabla \xi_1 + \alpha \xi_1 N_2 \right).
# $$
#
# ---
#
# ### Step 2: Substitute $ N_1 $ into the Second Equation
#
# Now substitute the expression for $ N_1 $ into the second equation:
#
# $$
# \frac{1}{D_{23}} N_2 - \beta \xi_2 \left( \frac{1}{D_{13}} + \alpha \xi_2 \right)^{-1} \left( -\nabla \xi_1 + \alpha \xi_1 N_2 \right) + \beta \xi_1 N_2 = -\nabla \xi_2.
# $$
#
# ---
#
# ### Step 3: Collect Terms for $ N_2 $
#
# Rearrange the terms to isolate $ N_2 $. Move everything with $ N_2 $ to the left side:
#
# $$
# \left( \frac{1}{D_{23}} + \beta \xi_1 - \frac{\beta \xi_2 \alpha \xi_1}{\frac{1}{D_{13}} + \alpha \xi_2} \right) N_2 = -\nabla \xi_2 + \frac{\beta \xi_2}{\frac{1}{D_{13}} + \alpha \xi_2} \nabla \xi_1.
# $$
#
# Thus:
#
# $$
# N_2 = -\left( \frac{1}{D_{23}} + \beta \xi_1 - \frac{\beta \xi_2 \alpha \xi_1}{\frac{1}{D_{13}} + \alpha \xi_2} \right)^{-1} \left( \nabla \xi_2 - \frac{\beta \xi_2}{\frac{1}{D_{13}} + \alpha \xi_2} \nabla \xi_1 \right).
# $$
#
# ---
#
# ### Step 4: Write the Final Equations for $ N_1 $ and $ N_2 $
#
# Now we can summarize the final expressions for $ N_1 $ and $ N_2 $.
#
# $$
# N_1 = -\left( \frac{1}{D_{13}} + \alpha \xi_2 \right)^{-1} \left( \nabla \xi_1 - \alpha \xi_1 N_2 \right),
# $$
#
# $$
# N_2 = -\left( \frac{1}{D_{23}} + \beta \xi_1 - \frac{\beta \xi_2 \alpha \xi_1}{\frac{1}{D_{13}} + \alpha \xi_2} \right)^{-1} \left( \nabla \xi_2 - \frac{\beta \xi_2}{\frac{1}{D_{13}} + \alpha \xi_2} \nabla \xi_1 \right).
# $$
#
# ---
#
#
# ### Step 5: Substitute when $D_{12}$ = $D_{13}$ alpha = 0:
#
# $$
# N_1 = -D_{12} \nabla \xi_1,
# $$
#
# $$
# N_2 = -\left( \frac{1}{D_{23}} + \beta \xi_1 \right)^{-1} \left( \nabla \xi_2 + \beta D_{12} \xi_2 \nabla \xi_1 \right).
# $$

# %%
D_12, D_13, D_23, D_21 = sp.symbols('D_12 D_13 D_23 D_21')
xi_1, xi_2, xi_3 = sp.symbols('xi_1 xi_2, xi_3')
alpha = 0
beta =(D_12**-1 - D_23**-1)  #((1/D_23) + (xi_1*(1/D_12 - 1/D_23)))**-1
N2 = sp.symbols('N2')

nabla_xi1, nabla_xi2 = sp.symbols(r'\nabla_xi1 \nabla_xi2')


grad_matrix = sp.Matrix([[nabla_xi1], [nabla_xi2]])


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

        gradients = sp.Matrix([[grad_C_x], [grad_C_y], [grad_C_z]])

    return gradients


# %%
A_grad = gradient_calc(A.sym)
B_grad = gradient_calc(B.sym)

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
B_matrix = create_B(3)

B_matrix

# %%
D_tilda = B_matrix**-1 

# %%
flux_matrix_sym = D_tilda * grad_matrix
flux_matrix_sym

# %%
flux_matrix = sp.simplify(flux_matrix_sym.subs({nabla_xi1:A_grad, nabla_xi2:B_grad, xi_1:A.sym[0], xi_2: B.sym[0], xi_3:C.sym[0], D_12:D12, D_13: D13, D_23:D23, D_21:D12}))

A_flux_term = flux_matrix[0]
    
B_flux_term = flux_matrix[1]


A_flux_star = A_flux_term.copy()
B_flux_star = B_flux_term.copy()

# %% [markdown]
# ### Setup timesteps
# $$
# \Delta t \leq \frac{\Delta x^2}{2 \max(D_{12}, D_{13}, D_{23})}
# $$

# %%
dt = mesh.get_min_radius()**2 / ( max(D12, D13, D23) )
dt

# %%
model_time = 0.
step = 0

# %%
### centre of domain
sample_spot = np.array([[0.72, 0.5]])
sample_spot


# %%
def solve_diff_eq(solver, u, u_star, flux_vector, flux_vector_star, dt):
    ### setup f0 term
    solver.f = -((u.sym[0] - u_star.sym[0]) / dt)

    
    

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
    
    # with mesh.access(u, u_star):
    #     u_star.data[...] = u.data[...]

# %% [markdown]
# ### Setup boundary conditions
# None are set as all boundaries are 0 flux, i.e. Neumann conditions

# %%
with mesh.access(A):
    cbar = plt.scatter(A.coords[:,0], A.coords[:,1], c=A.data[:,0])
    plt.colorbar(cbar)

    plt.scatter(sample_spot[:,0], sample_spot[:,1], marker='x', c='k', s=100)

# %%
A_vals     = []
B_vals     = []
time_arr   = []

# %%

while model_time < model_duration:


    
    if model_time + dt > model_duration:
        dt = model_duration - model_time

    if uw.mpi.rank ==0:
        print(f'step = {step}, time = {model_time}', flush=True)

    A_vals.append( uw.function.evaluate(A.sym, sample_spot) )
    B_vals.append( uw.function.evaluate(B.sym, sample_spot) )

    time_arr.append( model_time )

    # A_grad = gradient_calc(A.sym)
    # A_flux_term = flux1.subs({nabla_xi1:A_grad})

    # B_grad = gradient_calc(B.sym[0])
    # A_star_grad = gradient_calc(A_star.sym)
    # B_flux_term = flux2.subs({nabla_xi1:A_star_grad, nabla_xi2:B_grad, xi_1:A_star.sym[0], xi_2: B.sym[0]})


    flux_matrix = sp.simplify(flux_matrix_sym.subs({nabla_xi1:A_grad, nabla_xi2:B_grad, xi_1:A.sym[0], xi_2: B.sym[0], xi_3:C.sym[0], D_12:D12, D_13: D13, D_23:D23, D_21:D12}))

    A_flux_term = flux_matrix[0]
    
    B_flux_term = flux_matrix[1]


    
    
    ### solve the equation
    solve_diff_eq(A_diffusion, A, A_star, A_flux_term, A_flux_star, dt)
    solve_diff_eq(B_diffusion, B, B_star, B_flux_term, B_flux_star, dt)

    ### update history terms
    with mesh.access(A, A_star):
        A_star.data[...] = A.data[...]
    with mesh.access(B, B_star):
        B_star.data[...] = B.data[...]

    A_flux_star = A_flux_term.copy()
    B_flux_star = B_flux_term.copy()

    ### update time
    model_time += dt
    step += 1

# %%
with mesh.access(A):
    print(A.data[:,0].min(), A.data[:,0].max())

# %%
with mesh.access(A):
    cbar = plt.scatter(A.coords[:,0], A.coords[:,1], c=A.data[:,0])
    plt.colorbar(cbar)

    plt.scatter(sample_spot[:,0], sample_spot[:,1], marker='x', c='k', s=100)

# %%
with mesh.access(B):
    cbar = plt.scatter(B.coords[:,0], B.coords[:,1], c=B.data[:,0])
    plt.colorbar(cbar)

    plt.scatter(sample_spot[:,0], sample_spot[:,1], marker='x', c='k', s=100)

# %% [markdown]
# ### Compare with Geiser, 2015

# %%
Xi3 = np.array([[0.00639492, 0.70831986],
       [0.01188361, 0.68888466],
       [0.01894049, 0.67004506],
       [0.02756558, 0.65192932],
       [0.03775886, 0.63437252],
       [0.04873624, 0.61797971],
       [0.06206591, 0.60012816],
       [0.07853198, 0.58194212],
       [0.09578215, 0.56431721],
       [0.11303232, 0.54936992],
       [0.12923702, 0.53660775],
       [0.14686057, 0.52461217],
       [0.16366268, 0.5139544 ],
       [0.18242504, 0.50477874],
       [0.19998014, 0.49605567],
       [0.21757879, 0.48848252],
       [0.23519487, 0.4817143 ],
       [0.25347308, 0.47528344],
       [0.26969521, 0.47055575],
       [0.28745069, 0.4654418 ],
       [0.30403873, 0.46138703],
       [0.32160253, 0.45728104],
       [0.3388527 , 0.45376787],
       [0.35641651, 0.45026254],
       [0.37335304, 0.447031  ],
       [0.39060321, 0.44453099],
       [0.40785338, 0.44203097],
       [0.42510354, 0.43953095],
       [0.44235371, 0.43775461],
       [0.45913342, 0.43589053],
       [0.47685405, 0.43376772],
       [0.49410422, 0.43199138],
       [0.51135438, 0.43035978],
       [0.52860455, 0.42894528],
       [0.54585472, 0.42767552],
       [0.56310489, 0.42640575],
       [0.58035506, 0.42542546],
       [0.59760522, 0.4245899 ],
       [0.61485539, 0.42288593],
       [0.63210556, 0.4217609 ],
       [0.64935573, 0.42150429],
       [0.6666059 , 0.41951084],
       [0.68385607, 0.41983317],
       [0.70110623, 0.41776736],
       [0.7183564 , 0.41837916],
       [0.73560657, 0.41616861],
       [0.75285674, 0.41699752],
       [0.77010691, 0.41551065],
       [0.78735707, 0.4151093 ],
       [0.80460724, 0.41499742],
       [0.82185741, 0.41343819],
       [0.83910758, 0.41397762],
       [0.85635775, 0.41408284],
       [0.87360792, 0.41194467],
       [0.89085808, 0.41233936],
       [0.90810825, 0.41302353],
       [0.92535842, 0.41146429],
       [0.94260859, 0.410484  ],
       [0.95985876, 0.41073396],
       [0.97710892, 0.4112734 ],
       [0.9920068 , 0.41159124]])

# %%
Xi2 = np.array([[0.01226533, 0.18819946],
       [0.02465043, 0.18123445],
       [0.04322726, 0.17714486],
       [0.05971362, 0.17165274],
       [0.07696379, 0.16806721],
       [0.09421395, 0.16506061],
       [0.11146412, 0.1629948 ],
       [0.12871429, 0.16114609],
       [0.14596446, 0.16002106],
       [0.16321463, 0.15925788],
       [0.1804648 , 0.15929073],
       [0.19771496, 0.15874465],
       [0.21496513, 0.15935645],
       [0.2322153 , 0.1593893 ],
       [0.24946547, 0.15942216],
       [0.26687246, 0.16072899],
       [0.2839658 , 0.16107997],
       [0.30121597, 0.16241545],
       [0.31846614, 0.16295488],
       [0.33571631, 0.16436273],
       [0.35296648, 0.16540874],
       [0.37021664, 0.16623764],
       [0.38746681, 0.16757312],
       [0.40471698, 0.1686915 ],
       [0.42196715, 0.16937567],
       [0.43921732, 0.17092825],
       [0.45646749, 0.17182952],
       [0.47371765, 0.17265842],
       [0.49096782, 0.17413864],
       [0.50821799, 0.17496754],
       [0.52546816, 0.17586882],
       [0.54271833, 0.1772043 ],
       [0.55996849, 0.17774373],
       [0.57721866, 0.17915158],
       [0.59446883, 0.17954627],
       [0.611719  , 0.18073702],
       [0.62896917, 0.18105935],
       [0.64621934, 0.18246719],
       [0.6634695 , 0.18264479],
       [0.68071967, 0.18405263],
       [0.69796984, 0.18415786],
       [0.71522001, 0.18549334],
       [0.73247018, 0.18581567],
       [0.74972034, 0.18628273],
       [0.76697051, 0.18740111],
       [0.78422068, 0.18750633],
       [0.80147085, 0.18847997],
       [0.81872102, 0.18916414],
       [0.83597118, 0.189197  ],
       [0.85322135, 0.19009827],
       [0.87047152, 0.19085481],
       [0.88772169, 0.19088766],
       [0.90497186, 0.19120999],
       [0.92222203, 0.19232837],
       [0.93947219, 0.19257833],
       [0.95672236, 0.19261119],
       [0.97397253, 0.19293352],
       [0.98651811, 0.19426003]])

# %%
Xi1 = np.array([[0.00129828, 0.0795438 ],
       [0.0134518 , 0.12892185],
       [0.01972459, 0.1476409 ],
       [0.03253036, 0.17770317],
       [0.04638394, 0.20414465],
       [0.05736132, 0.22194394],
       [0.0699069 , 0.23987889],
       [0.08402067, 0.25773722],
       [0.09970264, 0.2750413 ],
       [0.11616871, 0.29123965],
       [0.13341888, 0.30581845],
       [0.15066905, 0.31851569],
       [0.16791922, 0.32962083],
       [0.18516939, 0.33877205],
       [0.20241955, 0.34734431],
       [0.21966972, 0.35446922],
       [0.23691989, 0.36065335],
       [0.25417006, 0.3658967 ],
       [0.27142023, 0.37070583],
       [0.2886704 , 0.37457419],
       [0.30592056, 0.37822544],
       [0.32317073, 0.38122538],
       [0.3404209 , 0.38393585],
       [0.35767107, 0.38606738],
       [0.37492124, 0.38805417],
       [0.3921714 , 0.38975149],
       [0.40942157, 0.3912317 ],
       [0.42667174, 0.39263955],
       [0.44392191, 0.39368556],
       [0.46117208, 0.39473156],
       [0.47842225, 0.3957052 ],
       [0.49567241, 0.396317  ],
       [0.51292258, 0.39700117],
       [0.53017275, 0.39775771],
       [0.54742292, 0.39800767],
       [0.56467309, 0.39847473],
       [0.58192325, 0.39886943],
       [0.59917342, 0.39962597],
       [0.61642359, 0.39973119],
       [0.63367376, 0.39976405],
       [0.65092393, 0.39979691],
       [0.66817409, 0.3999745 ],
       [0.68464016, 0.40065718],
       [0.70267443, 0.40069153],
       [0.7199246 , 0.40072438],
       [0.73717477, 0.40075724],
       [0.75442494, 0.4010072 ],
       [0.7716751 , 0.401619  ],
       [0.78892527, 0.40165186],
       [0.80617544, 0.40168472],
       [0.82342561, 0.40171758],
       [0.84067578, 0.40175043],
       [0.85792594, 0.40178329],
       [0.87517611, 0.40181615],
       [0.89242628, 0.40184901],
       [0.90967645, 0.40188186],
       [0.92692662, 0.40176999],
       [0.94417679, 0.40194758],
       [0.96142695, 0.40198044],
       [0.97867712, 0.40201329],
       [0.99435909, 0.40097478]])

# %%
plt.plot([], [], ' ', label="Analytical")
plt.scatter(Xi1[:,0], Xi1[:,1], marker='v', label=r'$\xi_1$', c='red', alpha=0.5)
plt.scatter(Xi2[:,0], Xi2[:,1], marker='^', label=r'$\xi_2$', c='blue', alpha=0.5)
plt.scatter(Xi3[:,0], Xi3[:,1], marker='d', label=r'$\xi_3$', c='green', alpha=0.5)

plt.plot([], [], ' ', label="UW")
plt.plot(time_arr, A_vals, label=r'$C_1$', c='red')
plt.plot(time_arr, B_vals, label=r'$C_2$', c='blue')

plt.plot(time_arr, 1-np.array(A_vals)-np.array(B_vals), label=r'$C_3$', c='green')

plt.legend(ncols=2)

# lines = plt.gca().get_lines()
# include = [0,1,2]
# legend1 = plt.legend([lines[i] for i in include],[lines[i].get_label() for i in include], loc=1)

plt.ylabel(r'$\xi$')
plt.xlabel('$t$')



plt.savefig('benchmark_2.pdf')

# %%
