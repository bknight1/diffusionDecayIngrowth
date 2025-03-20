# %%
import underworld3 as uw
import numpy as np
import math

from sympy import symbols, exp, lambdify
import sympy as sp

import os


if uw.mpi.size == 1:
    import matplotlib.pyplot as plt

# %%
csize = uw.options.getReal(name='csize', default = 0.01)

degree = uw.options.getInt(name='degree', default = 2)

plotFigs = uw.options.getBool(name='plotFigs', default = False) ### C

model_duration = uw.options.getReal(name='duration', default = 1500) ### Myr 

pulse_time = uw.options.getReal(name='pulse_time', default = 1000) ### Myr  

T_initial = uw.options.getReal(name='initial_temp', default = 850)  # Starting temperature
T_pulse   = uw.options.getReal(name='pulse_temp', default = 850)  # peak temperature of pulse
T_base    = uw.options.getReal(name='base_temp', default = 750)    # Final asymptotic temperature




# %%
outputPath = f'./output/isotropic_dual-growth_duration={int(model_duration)}_2nd_growth={int(pulse_time)}Myr_csize={csize}_Diffusion-Decay_test/'

os.makedirs(outputPath, exist_ok=True)

# %%
mesh_qdegree = degree

# if uw.mpi.rank ==0:
#     print(f'\ntest profile = {test_profile}\n\n', flush=True)

# %%
# Define time as a symbolic variable
t_sym = sp.symbols('t')

# Define the time_range range
time_arr = np.linspace(0, model_duration, 1001)

# Std to generate the width of the gaussian
std_dev0 = 40


initial_T_profile = T_base + (800 - T_base) * sp.exp(-((t_sym - 0) ** 2 / (2 * std_dev0 ** 2)))



std_dev = 120
Tprofile4 = T_base + (850 - T_base) * sp.exp(-((t_sym - pulse_time) ** 2 / (2 * std_dev ** 2)))

# Define the temperature as a piecewise function
T = sp.Piecewise(
    (initial_T_profile, (t_sym < 500)),  # Asymptotic decrease
    # (T_base, (t > 150) & (t < mu-450)),  # Constant 750°C between 100 and 1000 Myr
    (Tprofile4, (t_sym >= 500)),  # Gaussian peak
    # (T_base, (t > mu+150) & (t <= model_duration))  # Constant 750°C after 1100 Myr
)





# %%
# # Define the time_range range
time_arr = np.linspace(0, model_duration, 1001)


# %%
if uw.mpi.size == 1:
    temp_fn = sp.lambdify(t_sym, T, 'numpy')
    temp = temp_fn(time_arr)
    print(np.average(temp))
    plt.plot(time_arr, temp, label='temperature profile')



    plt.xlabel('Time [Myr]')

    plt.ylabel('Temperature [$\degree$C]')

    plt.xlim(0, model_duration)

    plt.ylim(550, 1100)

    plt.legend(loc='upper left')

    plt.grid()

    plt.ylim(700, 900)

        
    plt.savefig(f'{outputPath}temp_profiles_for_testing.pdf')




    # plt.close()

# %%
temp_profile = temp_fn

temp = temp_profile(time_arr)


# %%
if uw.mpi.size == 1:
    plt.plot(time_arr, temp, label=f'Temperature profile')
    plt.legend()
    plt.savefig(f'{outputPath}temp_profiles_testing.pdf')
    # plt.close()

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
current_ratio_U238_to_U235 = 137.818
half_life_U235 = 703.8e6  # Half-life of U-235 in years
half_life_U238 = 4.468e9  # Half-life of U-238 in years

def calculate_U238_U235_values(zircon_growth_time):


    # Calculate the remaining fraction of each isotope after 1 Ga
    remaining_fraction_U235 = 0.5 ** (zircon_growth_time / half_life_U235)
    remaining_fraction_U238 = 0.5 ** (zircon_growth_time / half_life_U238)


    # Calculate the original ratio based on remaining fractions and current ratio
    # Since we know the current ratio and the remaining fractions of each isotope, we can work backwards to find the original ratio.
    # The formula transforms as: original_ratio = current_ratio * (remaining_fraction_U238 / remaining_fraction_U235)
    original_ratio_U238_to_U235 = current_ratio_U238_to_U235 * ( remaining_fraction_U235 / remaining_fraction_U238)

    U238_N0 = original_ratio_U238_to_U235
    U235_N0 = 1.

    return U238_N0, U235_N0

# %%
model_duration

# %%
U238_N0, U235_N0 = calculate_U238_U235_values(model_duration*1e6)
U238_N0, U235_N0

# %%
# half_life_U235 = 703.8e6 * u.year # Half-life of U-235 in years
# half_life_U238 = 4.468e9 * u.year # Half-life of U-238 in years

# Decay constant of U238
lambda_U238 = np.log(2) / (half_life_U238 * u.year)
lambda_U235 = np.log(2) / (half_life_U235 * u.year)


lambda_U238_nd = nd(lambda_U238)

lambda_U235_nd = nd(lambda_U235)

# %%
from enum import Enum

class boundaries_2D(Enum):
    outerBoundary0 = 21
    outerBoundary1 = 22
    outerBoundary2 = 23
    outerBoundary3 = 24
    outerBoundary4 = 25
    outerBoundary5 = 26
    outerBoundary6 = 27
    outerBoundary7 = 28
    innerBoundary0 = 31
    innerBoundary1 = 32
    innerBoundary2 = 33
    innerBoundary3 = 34
    innerBoundary4 = 35
    innerBoundary5 = 36
    innerBoundary6 = 37
    innerBoundary7 = 38


# %%
# Points coordinates of mesh

### width (x) = 0.6 (60 micron), height (y) = 1. (100 micron)

points0 = [(-0.3, 0.35, 0), (-0.2, 0.5, 0), (0.2, 0.5, 0), (0.3, 0.35, 0), 
        (0.3, -0.35, 0), (0.2, -0.5, 0), (-0.2, -0.5, 0), (-0.3,-0.35, 0)]

inner_zircon = np.array(points0)


points1 = [(-0.6, 0.35, 0), (-0.2, 0.9, 0), (0.2, 0.9, 0), (0.6, 0.35, 0), 
        (0.6, -0.35, 0), (0.2, -0.9, 0), (-0.2, -0.9, 0), (-0.6,-0.35, 0)]

outer_zircon = np.array(points1)



# plt.plot(inner_zircon[:,0], inner_zircon[:,1])
# plt.plot(outer_zircon[:,0], outer_zircon[:,1])

# plt.gca().set_aspect('equal')

# %%
if uw.mpi.rank == 0:
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("BoxInsideBox")

    cellSize = csize

    outer_points = outer_zircon
    outer_point_ids = [gmsh.model.geo.addPoint(x, y, z, cellSize) for x, y, z in outer_points]

    # Define the corner points of the inner box
    inner_points = inner_zircon
    inner_point_ids = [gmsh.model.geo.addPoint(x, y, z, cellSize) for x, y, z in inner_points]

    # Define the lines connecting the points of the outer box
    outer_line_ids = [gmsh.model.geo.addLine(outer_point_ids[i], outer_point_ids[(i + 1) % len(outer_point_ids)]) for i in range(len(outer_point_ids))]

    # Define the lines connecting the points of the inner box
    inner_line_ids = [gmsh.model.geo.addLine(inner_point_ids[i], inner_point_ids[(i + 1) % len(inner_point_ids)]) for i in range(len(inner_point_ids))]

    # Create line loops for the inner box and the outer box
    inner_loop = gmsh.model.geo.addCurveLoop(inner_line_ids)
    outer_loop = gmsh.model.geo.addCurveLoop(outer_line_ids)

    # Create plane surfaces for the inner box and the outer box
    inner_surface = gmsh.model.geo.addPlaneSurface([inner_loop])
    outer_surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

    gmsh.model.geo.synchronize()

    # Add physical groups for the boundaries
    for i, line_id in enumerate(inner_line_ids):
        boundary_tag = getattr(boundaries_2D, f"innerBoundary{i}")
        gmsh.model.addPhysicalGroup(1, [line_id], tag=boundary_tag.value, name=boundary_tag.name)

    for i, line_id in enumerate(outer_line_ids):
        boundary_tag = getattr(boundaries_2D, f"outerBoundary{i}")
        gmsh.model.addPhysicalGroup(1, [line_id], tag=boundary_tag.value, name=boundary_tag.name)

    # gmsh.model.addPhysicalGroup(1, inner_line_ids, tag=boundaries_2D.innerBoundary.value)
    # gmsh.model.setPhysicalName(1, 5, boundaries_2D.innerBoundary.name)

    # gmsh.model.addPhysicalGroup(1, outer_line_ids, tag=boundaries_2D.outerBoundary.value)
    # gmsh.model.setPhysicalName(1, 5, boundaries_2D.outerBoundary.name)

    # Add physical groups for the surfaces
    gmsh.model.addPhysicalGroup(2, [outer_surface], tag=7)
    gmsh.model.setPhysicalName(2, 6, 'outer_surface')

    gmsh.model.addPhysicalGroup(2, [inner_surface], tag=8)
    gmsh.model.setPhysicalName(2, 7, 'inner_surface')

    # Add physical group for the surface
    gmsh.model.addPhysicalGroup(2, [outer_surface], 99999)
    gmsh.model.setPhysicalName(2, 99999, "Elements")


    # Generate a 2D mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh to a file
    # outputPath = "./meshes"
    gmsh.write(f'{outputPath}/dual_zircon_mesh_csize={csize}.msh')

    # Finalize the Gmsh session
    gmsh.finalize()

    # Print a message to indicate that the script has finished
    print("Mesh generation complete.", flush=True)

# %%
mesh = uw.discretisation.Mesh(
        f'{outputPath}/dual_zircon_mesh_csize={csize}.msh',
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

### Has to be same as the U and Pb mesh vars
zircon_regions  = uw.discretisation.MeshVariable("zircon_regions", mesh, 1, degree=degree)

### U238 to Pb206
Pb206 = uw.discretisation.MeshVariable("Pb206", mesh, 1, degree=degree)

Pb206_star = uw.discretisation.MeshVariable("Pb206_star", mesh, 1, degree=degree)

U238 = uw.discretisation.MeshVariable("U238", mesh, 1, degree=degree)

U238_star = uw.discretisation.MeshVariable("U238_star", mesh, 1, degree=degree)


### U235 to Pb207
Pb207 = uw.discretisation.MeshVariable("Pb207", mesh, 1, degree=degree)

Pb207_star = uw.discretisation.MeshVariable("Pb207_star", mesh, 1, degree=degree)

U235 = uw.discretisation.MeshVariable("U235", mesh, 1, degree=degree)

U235_star = uw.discretisation.MeshVariable("U235_star", mesh, 1, degree=degree)


# %%
### Label the inner and outer zircon regions
from matplotlib.patches import Circle, Polygon, Wedge
internal_zircon_polygon = Polygon(inner_zircon[:,0:2], closed=True)

with mesh.access(zircon_regions):
    zircon_regions.data[:,0] = internal_zircon_polygon.contains_points(zircon_regions.coords)

# %%
### swarm to store the actual point data
swarm = uw.swarm.Swarm(mesh)

Pb206_swarm = swarm.add_variable('Pb206_swarm')
Pb207_swarm = swarm.add_variable('Pb207_swarm')

U235_swarm = swarm.add_variable('U235_swarm')
U238_swarm = swarm.add_variable('U238_swarm')


swarm.add_particles_with_coordinates(Pb206.coords)

# %%
with mesh.access(Pb206, Pb206_star):
    Pb206.data[...] = 0.
    Pb206_star.data[...] = np.copy(Pb206.data[...])

with mesh.access(U238, U238_star, zircon_regions):
    U238.data[(zircon_regions.data[:,0] == 1)] = U238_N0
    U238_star.data[...] = np.copy(U238.data[...])

with mesh.access(Pb207, Pb207_star):
    Pb207.data[...] = 0.
    Pb207_star.data[...] = np.copy(Pb207.data[...])

with mesh.access(U235, U235_star, zircon_regions):
    U235.data[(zircon_regions.data[:,0] == 1)] = U235_N0
    U235_star.data[...] = np.copy(U235.data[...])

# %%
x_coords = np.arange(outer_zircon[:,0].min(), outer_zircon[:,0].max()+mesh.get_min_radius(), mesh.get_min_radius())
y_coords = np.zeros_like(x_coords)
sample_coords_x = np.column_stack([x_coords, y_coords])


# %%
Pb206_1D_x = uw.function.evalf(Pb206.sym[0], sample_coords_x)
## BC
Pb206_1D_x[0], Pb206_1D_x[-1] = 0., 0.

U238_1D_x = uw.function.evalf(U238.sym[0], sample_coords_x)
## BC
U238_1D_x[0], U238_1D_x[-1] = 0., 0.

# %%
Pb207_1D_x = uw.function.evalf(Pb207.sym[0], sample_coords_x)
## BC
Pb207_1D_x[0], Pb207_1D_x[-1] = 0., 0.

U235_1D_x = uw.function.evalf(U235.sym[0], sample_coords_x)
## BC
U235_1D_x[0], U235_1D_x[-1] = 0., 0.

# %%
Pb206_isotropicDiffusion = uw.systems.Poisson(mesh, u_Field=Pb206)

U238_isotropicDiffusion = uw.systems.Poisson(mesh, u_Field=U238)


Pb207_isotropicDiffusion = uw.systems.Poisson(mesh, u_Field=Pb207)

U235_isotropicDiffusion = uw.systems.Poisson(mesh, u_Field=U235)


# %%
### fix temp of top and bottom walls
### Pb and U boundaries set to 0

for boundary in mesh.boundaries:
    Pb206_isotropicDiffusion.add_dirichlet_bc(0., boundary.name)
    U238_isotropicDiffusion.add_dirichlet_bc(0., boundary.name)
    Pb207_isotropicDiffusion.add_dirichlet_bc(0., boundary.name)
    U235_isotropicDiffusion.add_dirichlet_bc(0., boundary.name)


# %%
for _solver in [Pb206_isotropicDiffusion, Pb207_isotropicDiffusion, U238_isotropicDiffusion, U235_isotropicDiffusion]:
    _solver.petsc_options['snes_rtol'] = 1e-12
    _solver.petsc_options['snes_atol'] = 1e-6
    
    ### see the SNES output
    _solver.petsc_options["snes_converged_reason"] = None


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
def calculate_decay_ingrowth(dt, parent, parent_star, daughter, daughter_star, parent_lambda):
    ### generic function that can be used for any radioactive decay and ingrowth

    ### Calculate decay
    parent_lost_fn  =  parent.sym[0] - (parent.sym[0] * np.exp(-parent_lambda*dt))

    parent_lost_data = uw.function.evalf(parent_lost_fn, parent.coords)

    ### Update parent values and copy to history term
    with mesh.access(parent, parent_star):
        parent.data[:,0] -= parent_lost_data
        parent_star.data[:,0] = np.copy(parent.data[:,0])
    
    ### Update daughter values and copy to history term
    with mesh.access(daughter, daughter_star):
        daughter.data[:,0] += parent_lost_data
        daughter_star.data[:,0] = np.copy(daughter.data[:,0])

    # return parent_lost_data


# %%
### Using the Possion solver, we can set:
### f0 through Pb_isotropicDiffusion.f
### f1 through the flux term (above)

# %%
### setup Pb terms
### get current temperature from profile
Temp = temp_profile(0) 

D_Pb_nd = nd( (D_Pb_fn(Temp)   *u.meter**2/u.second) )

D_U_nd = nd( (D_U_fn(Temp)   *u.meter**2/u.second) )

# %%
### setup flux term
# flux_vector = anistropic_diffusion_setup(Pb, D_Pb_nd, D_Pb_nd)

# Pb_isotropicDiffusion.constitutive_model =  uw.constitutive_models.DiffusionModel

# Pb_isotropicDiffusion.constitutive_model.Parameters.diffusivity = D_Pb_nd

Pb206_flux_vector = anistropic_diffusion_setup(Pb206, D_Pb_nd, D_Pb_nd)
Pb206_flux_vector_star = Pb206_flux_vector.copy()


Pb207_flux_vector = anistropic_diffusion_setup(Pb207, D_Pb_nd, D_Pb_nd)
Pb207_flux_vector_star = Pb207_flux_vector.copy()

# %%
# U_isotropicDiffusion.constitutive_model =  uw.constitutive_models.DiffusionModel

# U_isotropicDiffusion.constitutive_model.Parameters.diffusivity = D_Pb_nd

U238_flux_vector = anistropic_diffusion_setup(U238, D_U_nd, D_U_nd)
U238_flux_vector_star = U238_flux_vector.copy()


U235_flux_vector = anistropic_diffusion_setup(U235, D_U_nd, D_U_nd)
U235_flux_vector_star = U235_flux_vector.copy()

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

        print(nts)
    
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

step = 0
model_time = 0.



# %%
if uw.mpi.rank == 0: 
    dim_time = dim(model_time, u.megayear)
    print(f'step: {step}, time: {dim_time}\n\n', flush=True)

mesh.petsc_save_checkpoint(index=step, meshVars=[Pb206, Pb207, U235, U238, zircon_regions], outputPath=outputPath)   


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
def diffusion_decay(U, U_star, Pb, Pb_star, U_flux_vector_star, Pb_flux_vector_star, U_isotropicDiffusion, Pb_isotropicDiffusion, D_U_nd, D_Pb_nd, lambda_U_nd, dt):
    """Generic function for diffusion-decay of Uranium to Lead"""

     # Decay U to Pb
    calculate_decay_ingrowth(dt, U, U_star, Pb, Pb_star, lambda_U_nd)
    
    # Set up U flux vector
    U_flux_vector = anistropic_diffusion_setup(U, D_U_nd, D_U_nd)
    
    # Solve U diffusion
    solve_diff_eq(U_isotropicDiffusion, U, U_star, U_flux_vector, U_flux_vector_star, dt)
    
    # Update U flux history term
    U_flux_vector_star = U_flux_vector.copy()
    
   
    
    # Set up Pb flux vector
    Pb_flux_vector = anistropic_diffusion_setup(Pb, D_Pb_nd, D_Pb_nd)
    
    # Solve Pb diffusion
    solve_diff_eq(Pb_isotropicDiffusion, Pb, Pb_star, Pb_flux_vector, Pb_flux_vector_star, dt)
    
    # Update Pb flux history term
    Pb_flux_vector_star = Pb_flux_vector.copy()


# %%
while model_time < nd(pulse_time*u.megayear):

    # if step % 10 == 0:
    #     mesh.petsc_save_checkpoint(index=step, meshVars=[Pb], outputPath=outputPath)   

    ### get current temperature from profile
    Temp = temp_profile( dim(model_time,u.megayear).m ) 
    
    if uw.mpi.rank == 0:
        dim_time = dim(model_time, u.megayear)
        print(f'\nstep: {step}, temp: {Temp}, time: {dim_time}\n\n', flush=True)


    
    D_Pb_nd = nd( (D_Pb_fn(Temp)   *u.meter**2/u.second) )
    D_U_nd = nd( (D_U_fn(Temp)   *u.meter**2/u.second) )
    D_val = max(D_Pb_nd, D_U_nd)
    dt = min( (0.2 * (mesh.get_min_radius()**2 / D_val) ) , nd(50*u.megayear) )

    ### Finds where temp first and last exceeds 780 C and limits dt to capture the temperature pulse
    if (model_time + dt) < (nd(time_arr[temp > 800][0]*u.megayear)):
        dt = min(dt, (nd(time_arr[temp > 800][0]*u.megayear)) - model_time)
    elif ((model_time + dt) > (nd(time_arr[temp > 800][0]*u.megayear))) & ((model_time + dt) < nd(time_arr[temp > 800][-1]*u.megayear)):
        dt = min(dt, nd(5*u.megayear) )
        

    if (model_time + dt) > nd(pulse_time*u.megayear):
        dt = nd(pulse_time*u.megayear) - model_time



    ### U238 to Pb206
    diffusion_decay(U238, U238_star, Pb206, Pb206_star, U238_flux_vector_star, Pb206_flux_vector_star,
                    U238_isotropicDiffusion, Pb206_isotropicDiffusion, D_U_nd, D_Pb_nd, lambda_U238_nd, dt)

    ### U235 to Pb207
    diffusion_decay(U235, U235_star, Pb207, Pb207_star, U235_flux_vector_star, Pb207_flux_vector_star,
                    U235_isotropicDiffusion, Pb207_isotropicDiffusion, D_U_nd, D_Pb_nd, lambda_U235_nd, dt)



    # ### 1D line diffusion-decay-ingrwoth
    # ### determine amount lost along profile
    # U238_lost  =  U238_1D_x[1:-1] - (U238_1D_x[1:-1] * np.exp(-lambda_U238_nd*dt))
    
    # ### U238 diffusion and decay
    # U238_1D_x = diffusion_1D(sample_coords_x[:,0], U238_1D_x, 1*D_U_nd, dt)
    # U238_1D_x[1:-1]  -=  U238_lost
    # ### Pb207 diffusion and ingrowth
    # Pb206_1D_x[1:-1]  +=  U238_lost
    # Pb206_1D_x = diffusion_1D(sample_coords_x[:,0], Pb206_1D_x, 1*D_Pb_nd, dt)


    # ### 1D line diffusion-decay-ingrwoth
    # ### determine amount lost along profile
    # U235_lost  =  U235_1D_x[1:-1] - (U235_1D_x[1:-1] * np.exp(-lambda_U235_nd*dt))
    
    # ### U235 diffusion and decay
    # U235_1D_x = diffusion_1D(sample_coords_x[:,0], U235_1D_x, 1*D_U_nd, dt)
    # U235_1D_x[1:-1]  -=  U235_lost
    # ### Pb207 diffusion and ingrowth
    # Pb207_1D_x[1:-1]  +=  U235_lost
    # Pb207_1D_x = diffusion_1D(sample_coords_x[:,0], Pb207_1D_x, 1*D_Pb_nd, dt)





    step += 1
    model_time += dt


    





# %%
### get the second growth values
U238_N1, U235_N1 = calculate_U238_U235_values(model_duration*1e6 - pulse_time*1e6)

U238_N1, U235_N1

# %%
### update values in the outer zircon for the second growth
with mesh.access(Pb206, Pb206_star):
    Pb206.data[zircon_regions.data[:,0] == 0] = 0.
    Pb206_star.data[...] = np.copy(Pb206.data[...])

with mesh.access(U238, U238_star, zircon_regions):
    U238.data[(zircon_regions.data[:,0] == 0)] = U238_N1
    U238_star.data[...] = np.copy(U238.data[...])

with mesh.access(Pb207, Pb207_star):
    Pb207.data[zircon_regions.data[:,0] == 0] = 0.
    Pb207_star.data[...] = np.copy(Pb207.data[...])

with mesh.access(U235, U235_star, zircon_regions):
    U235.data[(zircon_regions.data[:,0] == 0)] = U235_N1
    U235_star.data[...] = np.copy(U235.data[...])

# %%
### Pb and U boundaries set to 0
for _solver in [Pb206_isotropicDiffusion, Pb207_isotropicDiffusion, U238_isotropicDiffusion, U235_isotropicDiffusion]:
    ### remove the BC from all boundaries
    _solver._reset() 
    ### reapply outer boundary condition
    for boundary in mesh.boundaries:
        if 'outer' in boundary.name:
            _solver.add_dirichlet_bc(0., boundary.name)

# %%
if uw.mpi.rank == 0: 
    dim_time = dim(model_time, u.megayear)
    print(f'step: {step}, time: {dim_time}\n\n', flush=True)

mesh.petsc_save_checkpoint(index=step, meshVars=[Pb206, Pb207, U235, U238, zircon_regions], outputPath=outputPath)   



# %%
total_time = nd(model_duration*u.megayear)

# %%
while model_time < total_time:

    # if step % 10 == 0:
    #     mesh.petsc_save_checkpoint(index=step, meshVars=[Pb], outputPath=outputPath)   

    ### get current temperature from profile
    Temp = temp_profile( dim(model_time,u.megayear).m ) 
    
    if uw.mpi.rank == 0:
        dim_time = dim(model_time, u.megayear)
        print(f'\nstep: {step}, temp: {Temp}, time: {dim_time}\n\n', flush=True)


    
    D_Pb_nd = nd( (D_Pb_fn(Temp)   *u.meter**2/u.second) )
    D_U_nd = nd( (D_U_fn(Temp)   *u.meter**2/u.second) )
    D_val = max(D_Pb_nd, D_U_nd)
    dt = min( (0.5 * (mesh.get_min_radius()**2 / D_val) ) , nd(2*u.megayear) )

    ### Finds where temp first and last exceeds 780 C and limits dt to capture the temperature pulse
    if (model_time + dt) < (nd(time_arr[temp > 800][0]*u.megayear)):
        dt = min(dt, (nd(time_arr[temp > 800][0]*u.megayear)) - model_time)
    elif ((model_time + dt) > (nd(time_arr[temp > 800][0]*u.megayear))) & ((model_time + dt) < nd(time_arr[temp > 800][-1]*u.megayear)):
        dt = min(dt, nd(5*u.megayear) )

    if (model_time + dt) > total_time:
        dt = total_time - model_time


    ### U238 to Pb206
    diffusion_decay(U238, U238_star, Pb206, Pb206_star, U238_flux_vector_star, Pb206_flux_vector_star,
                    U238_isotropicDiffusion, Pb206_isotropicDiffusion, D_U_nd, D_Pb_nd, lambda_U238_nd, dt)

    ### U235 to Pb207
    diffusion_decay(U235, U235_star, Pb207, Pb207_star, U235_flux_vector_star, Pb207_flux_vector_star,
                    U235_isotropicDiffusion, Pb207_isotropicDiffusion, D_U_nd, D_Pb_nd, lambda_U235_nd, dt)



    # ### 1D line diffusion-decay-ingrwoth
    # ### determine amount lost along profile
    # U238_lost  =  U238_1D_x[1:-1] - (U238_1D_x[1:-1] * np.exp(-lambda_U238_nd*dt))
    
    # ### U238 diffusion and decay
    # U238_1D_x = diffusion_1D(sample_coords_x[:,0], U238_1D_x, 1*D_U_nd, dt)
    # U238_1D_x[1:-1]  -=  U238_lost
    # ### Pb207 diffusion and ingrowth
    # Pb206_1D_x[1:-1]  +=  U238_lost
    # Pb206_1D_x = diffusion_1D(sample_coords_x[:,0], Pb206_1D_x, 1*D_Pb_nd, dt)


    # ### 1D line diffusion-decay-ingrwoth
    # ### determine amount lost along profile
    # U235_lost  =  U235_1D_x[1:-1] - (U235_1D_x[1:-1] * np.exp(-lambda_U235_nd*dt))
    
    # ### U235 diffusion and decay
    # U235_1D_x = diffusion_1D(sample_coords_x[:,0], U235_1D_x, 1*D_U_nd, dt)
    # U235_1D_x[1:-1]  -=  U235_lost
    # ### Pb207 diffusion and ingrowth
    # Pb207_1D_x[1:-1]  +=  U235_lost
    # Pb207_1D_x = diffusion_1D(sample_coords_x[:,0], Pb207_1D_x, 1*D_Pb_nd, dt)





    step += 1
    model_time += dt


# %%
# U_final = uw.function.evalf(U238.sym[0], sample_coords)
Pb206_final_x = uw.function.evalf(Pb206.sym[0], sample_coords_x[:])

U238_final_x = uw.function.evalf(U238.sym[0], sample_coords_x[:])


# U_final = uw.function.evalf(U238.sym[0], sample_coords)
Pb207_final_x = uw.function.evalf(Pb207.sym[0], sample_coords_x[:])

U235_final_x = uw.function.evalf(U235.sym[0], sample_coords_x[:])

# %%
if uw.mpi.rank == 0: 
    dim_time = dim(model_time, u.megayear)
    print(f'step: {step}, time: {dim_time}\n\n', flush=True)

mesh.petsc_save_checkpoint(index=step, meshVars=[Pb206, Pb207, U235, U238, zircon_regions], outputPath=outputPath)   



# %%
### save the swarm data too
with mesh.access(Pb207, Pb206, U235, U238):
    Pb206_data = Pb206.data[:,0]
    Pb207_data = Pb207.data[:,0]
    U235_data = U235.data[:,0]
    U238_data = U238.data[:,0]
    
with swarm.access(Pb207_swarm, Pb206_swarm, U235_swarm, U238_swarm):
    Pb207_swarm.data[:,0] = Pb207_data
    Pb206_swarm.data[:,0] = Pb206_data
    U235_swarm.data[:,0] = U235_data
    U238_swarm.data[:,0] = U238_data
    

swarm.write_timestep(f'swarm', f'fields', step, [Pb206_swarm, Pb207_swarm, U235_swarm, U238_swarm], outputPath, time=model_time)

# %%
# if uw.mpi.size == 1:
#     f, axs = plt.subplots(1, 2, figsize=(10,4), layout='constrained', sharex=True, sharey=False)
    
#     axs[0].plot(dim(sample_coords_x[:,0], u.micrometer).m, Pb206_final_x, c='red', label='$^{206}$Pb')
#     axs[0].plot(dim(sample_coords_x[:,0], u.micrometer).m, Pb206_1D_x, ls=':', c='k')
    
    
    
#     axs[0].plot(dim(sample_coords_x[:,0], u.micrometer).m, U238_1D_x, c='salmon',label='$^{238}$U')
#     axs[0].plot(dim(sample_coords_x[:,0], u.micrometer).m, U238_1D_x, c='k', ls=':')
    
#     ax0 = axs[0].twinx()
#     ax0.plot(dim(sample_coords_x[:,0], u.micrometer).m, Pb207_final_x, c='darkred', label='$^{207}$Pb')
#     ax0.plot(dim(sample_coords_x[:,0], u.micrometer).m, Pb207_1D_x, ls=':', c='k')
    
#     ax0.plot(dim(sample_coords_x[:,0], u.micrometer).m, U235_1D_x, c='darkorange',label='$^{235}$U')
#     ax0.plot(dim(sample_coords_x[:,0], u.micrometer).m, U235_1D_x, c='k', ls=':')
    
#     axs[1].plot(dim(sample_coords_x[:,0], u.micrometer).m, Pb207_1D_x/Pb206_1D_x, ls='-.', c='green', label=r'$\frac{{}^{207}\text{Pb}}{{}^{206}\text{Pb}}$')
#     ax1 = axs[1].twinx()
#     ax1.plot(dim(sample_coords_x[:,0], u.micrometer).m, U238_1D_x/Pb206_1D_x, ls='--', c='blue', label=r'$\frac{{}^{238}\text{U}}{{}^{206}\text{Pb}}$')
    
    
#     ax0.set_ylim(0, 0.8)
    
#     axs[0].set_ylim(0, 90)
    
    
#     axs[1].set_ylim(0.055, 0.0573)
    
#     ax1.set_ylim(10, 25)
    
#     ax1.set_yticks([10, 15, 20, 25])
#     # x1.set_yticklabels(labels)
    
#     # plt.plot(dim(x_coords, u.micrometer).m, U_final, c='k', ls=':')
#     axs[0].set_xlabel('coord [$\mu m$]')
#     axs[0].set_ylabel('$^{238}$U, $^{206}$Pb concentrations')
#     ax0.set_ylabel('$^{235}$U, $^{207}$Pb concentrations')
    
    
#     axs[1].set_xlabel('coord [$\mu m$]')
    
#     axs[1].set_ylabel(r'$\frac{{}^{207}\text{Pb}}{{}^{206}\text{Pb}}$', c='green')
#     ax1.set_ylabel(r'$\frac{{}^{238}\text{U}}{{}^{206}\text{Pb}}$', c='blue')
    
#     # plt.title(f'{Temp} $\degree $C')
    
#     axs[0].legend(bbox_to_anchor=(0.5,0.5), frameon=False)
#     ax0.legend(bbox_to_anchor=(0.75,0.5), frameon=False)
#     axs[1].legend(bbox_to_anchor=(0.54,0.5), frameon=False)
#     ax1.legend(bbox_to_anchor=(0.75,0.5), frameon=False)
    
#     axs[1].tick_params(axis='y', colors='green')
#     ax1.tick_params(axis='y', colors='blue')
    
    
    
#     plt.savefig(f'{outputPath}isotropic_diffusion-decay-U_Pb_profiles_temp-profile={test_profile}_duration={model_duration}_csize={csize}.pdf')

#     plt.close()


# %%

# %%



