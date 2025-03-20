# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Diffusion benchmark with 1D numerical solution
# - Benchmark comparison between 1D analytical solution and 2D UW numerical model.
#

# +
import underworld3 as uw
import numpy as np
import sympy
import math
import os

from scipy import special
# -

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt


# ### Set up variables of the model

# +
import sys


Tdegree = uw.options.getInt("Tdegree", default=2)

continuous =  uw.options.getBool("continuous", default=True)


start_time = uw.options.getReal("start_time", default=1e-4)

end_time = uw.options.getReal("end_time", default=2e-4)

kappa = uw.options.getReal("kappa", default=1.)

csize = uw.options.getReal("csize", default=0.025)


saveFigs = uw.options.getBool("saveFigs", default=False)

tolerance = uw.options.getReal("tolerance", default=1e-6)


outputPath = uw.options.getString('outputPath', default=f'./output/Diffusion_convergence_test/')

outputFile = uw.options.getString('outputFile', default=f'Diffusion_conv_test_HP_unstructuredSimplex_deg={Tdegree}_end_time={end_time}.txt')




# -


if uw.mpi.rank == 0:
    print(f'u degree: {Tdegree}, cell size: {csize}\n\n')

if uw.mpi.rank == 0:
    # checking if the directory
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

# ### Set up the mesh

# +
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 



model_length      = 100 * u.micrometer ### scale the mesh radius to the zircon radius

KL = model_length

scaling_coefficients  = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL

# +
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(1,1), cellSize=csize, qdegree=Tdegree) 



# -

# ### analytical solution

# +
# ### Determine analytical solution across the mesh
# and calculate the l1-norm & l2-norm

# u = 0.
# # t = dt
# t = start_time
# x0 = -0.3
# x1 = 0.3

u, t, x0, x1 = sympy.symbols('u, t, x0, x1')


U_a_x = 0.5 * ( sympy.erf( (x1  - mesh.X[0]+(u*t))  / (2*sympy.sqrt(kappa*t))) + sympy.erf( (-x0 + mesh.X[0]-(u*t))  / (2*sympy.sqrt(kappa*t))) )

U_a_x

# -

# ### Create mesh variables
# To be used in the solver

# +
# #### Create mesh vars
Pb = uw.discretisation.MeshVariable("Pb", mesh, 1, degree=Tdegree, continuous=continuous)

Pb_star = uw.discretisation.MeshVariable("Pb_star", mesh, 1, degree=Tdegree, continuous=continuous)
# -

start_condition = U_a_x.subs({u:0, t:start_time, x0 : 0.3, x1 : 0.6})
start_condition

with mesh.access(Pb, Pb_star):
    Pb.data[:,0] = uw.function.evaluate(start_condition, Pb.coords)
    Pb_star.data[:,0] = Pb.data[:,0]

# +
# if "uw_simplexbox" not in mesh.name:
#     with mesh.access(Pb, Pb_star):
#         Pb.data[(Pb.coords[:,0] > -0.1) & (Pb.coords[:,0] < 0.1) & (Pb.coords[:,1] > -0.3) & (Pb.coords[:,1] < 0.3)] = 1.
#         Pb_star.data[...] = Pb.data
# else:
#     with mesh.access(Pb, Pb_star):
#         Pb.data[(Pb.coords[:,0] >= -0.3) & (Pb.coords[:,0] <= 0.3)] = 1. # & (Pb.coords[:,1] > -0.3) & (Pb.coords[:,1] < 0.3)] = 1.
#         Pb_star.data[...] = Pb.data
# -

# #### Create the diffusion solver using the Poisson solver

diffusion = uw.systems.Poisson(mesh, u_Field=Pb)

# ### Set up properties of the diffusion solver
# - Constitutive model (Diffusivity)
# - Boundary conditions

# +
diffusion.constitutive_model = uw.constitutive_models.DiffusionModel
diffusion.constitutive_model.Parameters.diffusivity = kappa


# +
# for boundary in mesh.boundaries:
#     diffusion.add_dirichlet_bc(0., boundary.name)


diffusion.add_dirichlet_bc(0., mesh.boundaries.Left.name)
diffusion.add_dirichlet_bc(0., mesh.boundaries.Right.name)
# -


xmin, xmax = mesh.data[:,0].min(), mesh.data[:,0].max()
xmin, xmax

# +
### y coords to sample
### nsamples needs to be high enough to resolve edges
nsamples = 201
sample_x = np.linspace(xmin, xmax, nsamples) ### get the x coords from the mesh

### x coords to sample
sample_y = np.zeros_like(sample_x)

sample_points = np.empty((sample_x.shape[0], 2))
sample_points[:, 0] = sample_x
sample_points[:, 1] = sample_y ###  across centre of box

# ### get the initial temp profile
# U_init = uw.function.evaluate(Pb.sym[0], sample_points)

# U_init[0], U_init[-1] = 0., 0. ### BC
# -

Pb_initial_profile = uw.function.evaluate(Pb.sym, sample_points)
Pb_analytical_profile = uw.function.evaluate(start_condition, sample_points)

# ### Solver loop

# +
diffusion.petsc_options["snes_rtol"]   = tolerance*1e-4
diffusion.petsc_options["snes_atol"]   = tolerance

diffusion.petsc_options["ksp_atol"]    = tolerance
diffusion.petsc_options["snes_max_it"] = 100

diffusion.petsc_options["snes_monitor_short"] = None

# if uw.mpi.size == 1:
#     diffusion.petsc_options['pc_type'] = 'lu'


# +
dt_diff = mesh.get_min_radius()**2 / kappa

dt =  dt_diff

time = start_time

step = 0

# +
### setup f0 term
diffusion.f = - ((Pb.sym[0] - Pb_star.sym[0]) / dt)

theta = 0

### setup flux term and history term
flux_vector = diffusion.constitutive_model.flux

diffusion.flux = theta*flux_vector + ((1-theta)*flux_vector)
# -

diffusion.f

diffusion.flux

diffusion.view()

diffusion.solve()

# +
while  time < end_time:

    if time + dt > end_time:
        dt = end_time - time
    
    print(f'\nstep: {step}, time: {time} \n', flush=True)
    ### setup f0 term
    diffusion.f = - ((Pb.sym[0] - Pb_star.sym[0]) / dt)
    
    theta = 0
    
    ### setup flux term and history term
    flux_vector = diffusion.constitutive_model.flux
    
    diffusion.flux = theta*flux_vector + ((1-theta)*flux_star)

    ### solve
    diffusion.solve()

    ### update history terms

    flux_star = diffusion.constitutive_model.flux.copy()

    with mesh.access(Pb, Pb_star):
        Pb_star.data[...] = Pb.data[...]

    time += dt

    step += 1

### print when finished
print(f'\nstep: {step}, time: {time} \n', flush=True)
# -


# ### Determine numerical solution across the mesh
# and calculate the l2-norm

final_condition = U_a_x.subs({u:0, t:end_time, x0 : 0.3, x1 : 0.6})

l2_fn = sympy.sqrt((Pb.sym[0]-final_condition)**2)

# +
I = uw.maths.Integral(mesh, l2_fn)

l2_norm = ( I.evaluate() )

print(f'\nL2 norm: {l2_norm}\n', flush=True)

# +
# L2 norm: 0.0031254634498052423
# -

U_a_final = uw.function.evaluate(final_condition, sample_points)
U_n_final = uw.function.evaluate(Pb.sym, sample_points)

# +
if saveFigs == True & uw.mpi.size == 1:
    ### only works if run in serial
    L2_grid_data = uw.function.evaluate( sympy.sqrt((final_condition - Pb.sym[0])**2), Pb.coords)
    from matplotlib import rc
    
    # Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':10})
    
    # Set the font used for MathJax
    rc('mathtext',**{'default':'regular'})
    rc('figure',**{'figsize':(8,6)})
    
    import matplotlib
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'Arial'
    
    f, ax = plt.subplots(1, 1,  sharey=False, sharex=True, figsize=(5, 4), layout='constrained')
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    
    
    ax.scatter(Pb.coords[:,0], Pb.coords[:,1], c=uw.function.evaluate(start_condition, Pb.coords), s=0.8)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.savefig(f'{outputPath}HP_results-deg={Tdegree}_csize={csize}_initial_condition.pdf')
    
    
    plt.close()
    
    f, ax = plt.subplots(1, 1,  sharey=False, sharex=True, figsize=(5, 4), layout='constrained')
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    
    with mesh.access(Pb):
        ax.scatter(Pb.coords[:,0], Pb.coords[:,1], c=Pb.data, s=0.8)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig(f'{outputPath}HP_results-deg={Tdegree}_csize={csize}_final_condition.pdf')
    
    
    plt.close()
    
    f, ax = plt.subplots(1, 1,  sharey=False, sharex=True, figsize=(5, 4), layout='constrained')
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    
    cbar = ax.scatter(Pb.coords[:,0], Pb.coords[:,1], c=L2_grid_data,  s=0.8, vmin=0, vmax=3e-3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(cbar, ax=ax, shrink=0.6, format='%.0e')
    
    
    plt.savefig(f'{outputPath}HP_results-deg={Tdegree}_csize={csize}_L2-norm_box.pdf')
    
    plt.close()




# +
def diffusion_1D(sample_points, T0, diffusivity, time_1D):
    x = sample_points
    T = T0.copy()
    k = diffusivity
    time = time_1D



    dx = sample_points[1] - sample_points[0]

    dt_dif = (dx**2 / k)

    dt = 0.1 * dt_dif


    if time > 0:

        """ determine number of its """
        nts = math.ceil(time / dt)

    
        """ get dt of 1D model """
        final_dt = time / nts

    
        for _ in range(nts):
            qT = -k * np.diff(T) / dx
            dTdt = -np.diff(qT) / dx
            T[1:-1] += dTdt * final_dt

    
    return T


# U_1D = diffusion_1D(x_coords, U_init, D_U_nd, end_time)
Pb_1D = diffusion_1D(sample_points[:,0], Pb_analytical_profile, kappa, (end_time - start_time))
# -

if uw.mpi.size == 1:
    f, ax = plt.subplots(1, 1,  sharey=False, sharex=True, figsize=(5, 4), layout='constrained')
    
    
    # plt.plot(sample_points[:,0], Pb_initial_profile, c='darkblue')
    ax.plot(sample_points[:,0], Pb_analytical_profile, ls='--', c='blue', label='inital profile')
    
    
    ax.plot(sample_points[:,0], U_a_final, c='k', label='analytical solution')
    ax.plot(sample_points[:,0], U_n_final, ls=(0, (5, 5)), c='red', label='UW solution')
    ax.plot(sample_points[:,0], Pb_1D, ls=':', c='lightgreen', label='1D FD solution')
    
    
    # plt.plot(sample_points[:,0], Pb_initial_profile, c='darkblue')
    
    # plt.plot(sample_points[:,0], Pb_analytical_profile, ls=':', c='red')
    ax.legend()
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.savefig(f'{outputPath}HP_profile_comp-deg={Tdegree}_csize={csize}-time={end_time}_1D_sol.pdf')



# +
### Create columns of file if it doesn't exist
try:
    with open(f'{outputPath}{outputFile}', 'x') as f:
        f.write(f'Tdegree,cell size,L2_norm')
except:
    pass


results = np.column_stack([Tdegree, mesh.get_min_radius(), l2_norm ])


### Append the data
with open(f'{outputPath}{outputFile}', 'a') as f:
    for i, item in enumerate(results.flatten()):
        if i == 0:
            f.write(f'\n{item}')
        else:
            f.write(f',{item}')
