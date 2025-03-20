# %%
import underworld3 as uw
import numpy as np
import math

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt

# %%
csize = uw.options.getReal(name='csize', default = 0.01)

degree = uw.options.getInt(name='degree', default = 2)


### Temp to remain constant for model run
Temp_start = uw.options.getReal(name='tempStart', default = 850) ### C
Temp_end = uw.options.getReal(name='tempEnd', default = 750) ### C

model_duration = uw.options.getReal(name='duration', default = 500) ### Myr 


# %%
import os

outputPath = f'./output/isotropic_decreasingT_duration={model_duration}_csize={csize}_Diffusion_test/'

os.makedirs(outputPath, exist_ok=True)

# %%
mesh_qdegree = degree
mesh_qdegree

# %%
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 

diffusive_rate    = 1e-22 * u.meter**2 /u.second


model_length      = 100 * u.micrometer ### scale the mesh radius to the zircon radius



KL = model_length
Kt = model_length**2 / diffusive_rate


scaling_coefficients  = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt

scaling_coefficients

# %% [markdown]
# | Variable | Symbol            | units | U | Pb | 
# | :---------- | :-------: | :-------: | :------: |  ------: | 
# | Pre-exponent| $D_0$   | $\text{m}^2\, \text{s}^{-1}$ |  1.63   | 0.11 |
# | Activation energy | $E_a$  | $\text{kJ}\, \text{mol}^{-1}$ |  726 $\pm$ 83    |  550 $\pm$ 30  |
# | Gas constant | $R$  | $\text{J}\, \text{mol}^{-1}\, \text{K}^{-1}$ |  8.314    | 8.314 | 
# | Reference | |  | [Cherniak and Watson, 1997](http://link.springer.com/10.1007/s004100050287) | [Cherniak and Watson, 2001](https://www.sciencedirect.com/science/article/pii/S0009254100002333) | [Cherniak and Watson, 2007](https://www.sciencedirect.com/science/article/pii/S0009254107002148) | 

# %%
from sympy import symbols, exp, lambdify
import sympy as sp

# Define the symbols
# R = 8.314  # Gas constant in J/(mol*K)
D, Ea, R, T = symbols('D Ea R T') # Temperature in Kelvin

D_sym = D * exp(-Ea / (R * (T+ 273.15) ) )

# %%
# Activation energy, taking the mean value for the sake of this example
Ea_U = 726e3 # Activation energy in J/mol

# Pre-exponential factor
D0_U = 10**0.212 # m^2/s

# The diffusivity equation
D_U_exp = D_sym.subs({D:D0_U, Ea:Ea_U, R:8.314 }) #D0 * exp(-Ea / (R * (T+ 273.15) ) )

# Convert the sympy expression to a numpy-compatible function
D_U_fn = lambdify((T), D_U_exp, 'numpy')

D_U_exp

# %%
# Activation energy, taking the mean value for the sake of this example
Ea_Pb = 550e3 # Activation energy in J/mol

# Pre-exponential factor
D0_Pb = 0.11 # m^2/s

# The diffusivity equation
D_Pb_exp = D_sym.subs({D:D0_Pb, Ea:Ea_Pb, R:8.314 }) # D0 * exp(-Ea / (R * (T+ 273.15) ) )

# Convert the sympy expression to a numpy-compatible function
D_Pb_fn = lambdify((T), D_Pb_exp, 'numpy')



# %%
from enum import Enum

class boundaries_2D(Enum):
    Boundary0 = 11
    Boundary1 = 12
    Boundary2 = 13
    Boundary3 = 14
    Boundary4 = 15
    Boundary5 = 16
    Boundary6 = 17
    Boundary7 = 18

    
# Points coordinates

### width (x) = 0.6 (60 micron), height (y) = 1. (100 micron)

points = [(-0.3, 0.35, 0), (-0.2, 0.5, 0), (0.2, 0.5, 0), (0.3, 0.35, 0), 
          (0.3, -0.35, 0), (0.2, -0.5, 0), (-0.2, -0.5, 0), (-0.3,-0.35, 0)]

points_array = np.array(points)


import gmsh

gmsh.initialize()
gmsh.model.add("box_with_circ")

cellSize = csize


# Create points
point_ids = [gmsh.model.geo.addPoint(x, y, z, meshSize=cellSize) for x, y, z in points]

# Create lines by connecting consecutive points and closing the loop
line_ids = [gmsh.model.geo.addLine(point_ids[i], point_ids[(i + 1) % len(point_ids)]) for i in range(len(point_ids))]

cl = gmsh.model.geo.addCurveLoop(line_ids)
surface = gmsh.model.geo.addPlaneSurface([cl])

gmsh.model.geo.synchronize()

# Adding physical groups for lines
for i, line_id in enumerate(line_ids):
    boundary_tag = getattr(boundaries_2D, f"Boundary{i}")
    gmsh.model.addPhysicalGroup(1, [line_id], tag=boundary_tag.value, name=boundary_tag.name)

# Add physical group for the surface
gmsh.model.addPhysicalGroup(2, [surface], 99999)
gmsh.model.setPhysicalName(2, 99999, "Elements")

gmsh.model.mesh.generate(2)
gmsh.write(f'{outputPath}zircon_mesh.msh')

gmsh.finalize()

# %%
mesh = uw.discretisation.Mesh(
        f'{outputPath}zircon_mesh.msh',
        degree=1,
        qdegree=mesh_qdegree,
        boundaries=boundaries_2D,
        boundary_normals=None,
        coordinate_system_type=uw.coordinates.CoordinateSystemType.CARTESIAN,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        refinement=None,
        refinement_callback=None,
    )

mesh.dm.view()

# %%
# mesh.vtk(f"{outputPath}zircon_mesh.vtk")

# import pyvista as pv
# pvmesh = pv.read(f"{outputPath}zircon_mesh.vtk")

# plotter = pv.Plotter()
# plotter.add_mesh(pvmesh, show_edges=True)

# plotter.show_bounds(
#             # grid='back',
#             location='outer',
#             all_edges=True,
#             )

# plotter.view_xy()  # if mesh_2D is on the xy plane.

# plotter.save_graphic(f'{outputPath}mesh_geom.pdf')
# plotter.show()

# # plotter.screenshot(f'{outputDir}{meshname}.png')  

# %%
# #### Create mesh vars

Pb = uw.discretisation.MeshVariable("Pb", mesh, 1, degree=degree)

Pb_star = uw.discretisation.MeshVariable("Pb_star", mesh, 1, degree=degree)

U = uw.discretisation.MeshVariable("U", mesh, 1, degree=degree)

U_star = uw.discretisation.MeshVariable("U_star", mesh, 2, degree=degree)

# %%
with mesh.access(Pb, Pb_star):
    Pb.data[...] = 1.
    Pb_star.data[...] = Pb.data[...]

with mesh.access(U, U_star):
    U.data[...] = 1.
    U_star.data[...] = U.data[...]

# %%

x_coords = np.arange(points_array[:,0].min(), points_array[:,0].max()+mesh.get_min_radius(), mesh.get_min_radius())
y_coords = np.zeros_like(x_coords)
sample_coords_x = np.column_stack([x_coords, y_coords])

# %%
Pb_init_x = uw.function.evalf(Pb.sym[0], sample_coords_x)
## BC
Pb_init_x[0], Pb_init_x[-1] = 0., 0.

U_init_x = uw.function.evalf(U.sym[0], sample_coords_x)
## BC
U_init_x[0], U_init_x[-1] = 0., 0.

U_profile = U_init_x.copy()
Pb_profile = Pb_init_x.copy()


# %%
Pb_isotropicDiffusion = uw.systems.Poisson(mesh, u_Field=Pb)

U_isotropicDiffusion = uw.systems.Poisson(mesh, u_Field=U)


# %%
### fix temp of top and bottom walls
### Pb and U boundaries set to 0
# # for _solver in solvers:
# for _solver in [Pb_isotropicDiffusion, U_isotropicDiffusion]:
#     for boundary in mesh.boundaries:
#         _solver.add_dirichlet_bc(0., boundary.name)

for boundary in mesh.boundaries:
    Pb_isotropicDiffusion.add_dirichlet_bc(0., boundary.name)

for boundary in mesh.boundaries:
    U_isotropicDiffusion.add_dirichlet_bc(0., boundary.name)

# %%
# for _solver in [Pb_isotropicDiffusion, U_isotropicDiffusion]:
Pb_isotropicDiffusion.petsc_options['snes_rtol'] = 1e-12
Pb_isotropicDiffusion.petsc_options['snes_atol'] = 1e-6

### see the SNES output
Pb_isotropicDiffusion.petsc_options["snes_converged_reason"] = None


U_isotropicDiffusion.petsc_options['snes_rtol'] = 1e-12
U_isotropicDiffusion.petsc_options['snes_atol'] = 1e-6

### see the SNES output
U_isotropicDiffusion.petsc_options["snes_converged_reason"] = None

# %%
step = 0
model_time = 0.



# %%
def anistropic_diffusion_setup(U_sym, Dx, Dy,):
    ### Creates an anistropic diffusion flux term
    ### determine the flux of the model
    x, y = mesh.X

    grad_T_x = sp.diff(U_sym.sym[0], x)
    grad_T_y = sp.diff(U_sym.sym[0], y)

    flux_x = Dx * grad_T_x
    flux_y = Dy * grad_T_y

    # Define the heat flux vector q as a matrix
    flux_vector = sp.Matrix([flux_x, flux_y])


    return flux_vector

# %%
### Using the Possion solver, we can set:
### f0 through Pb_isotropicDiffusion.f
### f1 through the flux term (above)

# %%
### setup Pb terms
Temp = Temp_start
D_Pb_nd = nd( (D_Pb_fn(Temp)   *u.meter**2/u.second) )

D_U_nd = nd( (D_U_fn(Temp)   *u.meter**2/u.second) )

# %%
### setup flux term
# flux_vector = anistropic_diffusion_setup(Pb, D_Pb_nd, D_Pb_nd)

# Pb_isotropicDiffusion.constitutive_model =  uw.constitutive_models.DiffusionModel

# Pb_isotropicDiffusion.constitutive_model.Parameters.diffusivity = D_Pb_nd

Pb_flux_vector = anistropic_diffusion_setup(Pb, D_Pb_nd, D_Pb_nd)
Pb_flux_vector_star = Pb_flux_vector.copy()

# %%
# U_isotropicDiffusion.constitutive_model =  uw.constitutive_models.DiffusionModel

# U_isotropicDiffusion.constitutive_model.Parameters.diffusivity = D_Pb_nd

U_flux_vector = anistropic_diffusion_setup(U, D_U_nd, D_U_nd)
U_flux_vector_star = U_flux_vector.copy()

# %%
def diffusion_1D(sample_points, T0, diffusivity, time_1D):
    x = sample_points
    T = T0.copy()
    k = diffusivity
    time = time_1D

    dx = sample_points[1] - sample_points[0]

    dt_dif = (dx**2 / k)

    dt = 0.2 * dt_dif


    if time > 0:

        """ determine number of its """
        nts = math.ceil(time / dt)

        # print(nts)
    
        """ get dt of 1D model """
        final_dt = time / nts

    
        for _ in range(nts):
            qT = -k * np.diff(T) / dx
            dTdt = -np.diff(qT) / dx
            T[1:-1] += dTdt * final_dt

    
    return T


# U_1D = diffusion_1D(x_coords, U_init, D_U_nd, total_time)
# Pb_1D = diffusion_1D(x_coords, Pb_init, D_Pb_nd, dt)

# %%
total_time = nd(model_duration*u.megayear)

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
while model_time < total_time:

    # if step % 10 == 0:
    #     mesh.petsc_save_checkpoint(index=step, meshVars=[Pb], outputPath=outputPath)   
          
    if uw.mpi.rank == 0:
        dim_time = dim(model_time, u.megayear)
        print(f'\nstep: {step}, time: {dim_time}\n\n', flush=True)

    Tgrad = (Temp_start - Temp_end)/model_duration
    Temp = Temp_start - (Tgrad*dim(model_time,u.megayear).m)

    D_Pb_nd = nd( (D_Pb_fn(Temp)   *u.meter**2/u.second) )
    D_U_nd = nd( (D_U_fn(Temp)   *u.meter**2/u.second) )
    D_val = max(D_Pb_nd, D_U_nd)

    ### reduce dt to max 2 Myrs to capture the change in temperature over time
    dt = min( (0.5 * (mesh.get_min_radius()**2 / D_val) ) , nd(2*u.megayear) )

    if (model_time + dt) > total_time:
        dt = total_time - model_time


    Pb_flux_vector = anistropic_diffusion_setup(Pb, D_Pb_nd, D_Pb_nd)
    ### solve Pb diffusion
    solve_diff_eq(Pb_isotropicDiffusion, Pb, Pb_star, Pb_flux_vector, Pb_flux_vector_star, dt)
    ### update Pb flux history term
    Pb_flux_vector_star = Pb_flux_vector.copy()

    U_flux_vector = anistropic_diffusion_setup(U, D_U_nd, D_U_nd)
    ### solve U diffusion
    solve_diff_eq(U_isotropicDiffusion, U, U_star, U_flux_vector, U_flux_vector_star, dt)
    ### update U flux history term
    U_flux_vector_star = anistropic_diffusion_setup(U, D_U_nd, D_U_nd)


    ### Do the 1D solution to compare to at the end

    U_profile = diffusion_1D(x_coords, U_profile, D_U_nd, dt)

    Pb_profile = diffusion_1D(x_coords, Pb_profile, D_Pb_nd, dt)


    step += 1
    model_time += dt


    





# %%
if uw.mpi.rank == 0: 
    dim_time = dim(model_time, u.megayear)
    print(f'step: {step}, time: {dim_time}\n\n', flush=True)

# mesh.petsc_save_checkpoint(index=step, meshVars=[Pb], outputPath=outputPath)   

# %%
# U_final = uw.function.evalf(U238.sym[0], sample_coords)
Pb_final_x = uw.function.evalf(Pb.sym[0], sample_coords_x[:])

U_final_x = uw.function.evalf(U.sym[0], sample_coords_x[:])


# %%
# Pb_1D_x = diffusion_1D(sample_coords_x[:,0], Pb_init_x, 1*D_Pb_nd, total_time)

# U_1D_x = diffusion_1D(sample_coords_x[:,0], Pb_init_x, 1*D_U_nd, total_time)

# %%
plt.plot(dim(sample_coords_x[:,0], u.micrometer).m, Pb_profile, c='red', label='Pb')
plt.plot(dim(sample_coords_x[:,0], u.micrometer).m, Pb_final_x, ls=':', c='k')


plt.plot(dim(sample_coords_x[:,0], u.micrometer).m, U_profile, c='salmon',label='U')
plt.plot(dim(sample_coords_x[:,0], u.micrometer).m, U_profile, c='k', ls=':')
# plt.plot(dim(x_coords, u.micrometer).m, U_final, c='k', ls=':')

plt.xlabel('coord [$\mu m$]')
plt.ylabel('concentration')

plt.title(f'From {Temp_start} $\degree $C to {Temp_end} $\degree $C')

plt.legend()


# plt.savefig(f'{outputPath}isotropic_diffusion-U_Pb_profiles-Tstart{Temp_start}_Tend={Temp_end}C_{model_duration}Myr_csize={csize}.pdf')

# %%



