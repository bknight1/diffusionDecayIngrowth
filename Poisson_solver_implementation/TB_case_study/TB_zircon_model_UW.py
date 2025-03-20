# %%
import underworld3 as uw
import numpy as np
import math

from scipy import interpolate

from sympy import symbols, exp, lambdify
import sympy as sp

import os

### for reloading the model
import pyvista as pv
from natsort import natsorted
from glob import glob
import re 


if uw.mpi.size == 1:
    import matplotlib.pyplot as plt

# %%
### don't need to be defined but can be
plotFigs = uw.options.getBool(name='plotFigs', default = False)

degree = uw.options.getInt(name='degree', default = 1)

csize = uw.options.getReal(name='csize', default = 0.01)

### These are used for scaling in the model
ref_length = uw.options.getReal(name='ref_length', default = 100.) ### in microns
ref_D = uw.options.getReal(name='ref_D', default = 1e-22) ### in u.meter**2 /u.second

### when to take timestep for restarting
timestep_interval = uw.options.getReal(name='timestep_interval', default = 200) 

### where to start the model from along the Temp-time path
start_time = uw.options.getReal(name='start_time', default=0)

### diffusion type, fast for radiation-damaged zircon, slow for not damaged/below alpha threshold
diffusion_type = uw.options.getString(name='diffusion_type', default = 'slow') ### in u.meter**2 /u.second


# %%
### need to be defined in the options

mesh_file = uw.options.getString(name='mesh_file')
output_path = uw.options.getString(name='output_path')


### use these to create a 1D line interp for getting the temp at a given time along the T-t path
time_arr = uw.options.getRealArray(name='time_arr')
temp_arr = uw.options.getRealArray(name='temp_arr')


model_duration = time_arr.max()


# print(time_arr)


# %%
### create the interpolation of time to temp along the T-t path
temp_profile = interpolate.interp1d(time_arr, temp_arr)

# %%
timestep_dir = f'{output_path}timesteps/'

restart_model = False

if os.path.exists(output_path):
    restart_model = True

# ### make dirs if don't exist
os.makedirs(timestep_dir, exist_ok=True)




# %%
mesh_qdegree = degree


# %%
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 


diffusive_rate    = ref_D * u.meter**2 /u.second


model_length      = ref_length * u.micrometer ### scale the mesh radius to the zircon radius



KL = model_length
Kt = model_length**2 / diffusive_rate


scaling_coefficients  = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt

scaling_coefficients

# %% [markdown]
# | Variable | Symbol            | units | U | Pb (slow) | Pb (fast) |
# | :---------- | :-------: | :-------: | :------: |  :------: |   ------: | 
# | Pre-exponent| $D_0$   | $\text{m}^2\, \text{s}^{-1}$ |  1.63   | 0.11 |  1.27 ${\times}10^{-8}$ |
# | Activation energy | $E_a$  | $\text{kJ}\, \text{mol}^{-1}$ |  726 $\pm$ 83    |  550 $\pm$ 30  |  228 $\pm$ 7.15 |
# | Gas constant | $R$  | $\text{J}\, \text{mol}^{-1}\, \text{K}^{-1}$ |  8.314    | 8.314 |    8.314 |
# | Reference | |  | [Cherniak and Watson, 1997](http://link.springer.com/10.1007/s004100050287) | [Cherniak and Watson, 2001](https://www.sciencedirect.com/science/article/pii/S0009254100002333) | [Cherniak et al., 1991](https://linkinghub.elsevier.com/retrieve/pii/001670379190137T) | |   

# %%
# Define the symbols
# R = 8.314  # Gas constant in J/(mol*K)
D, Ea, R, T = symbols('D Ea R T') # Temperature in Kelvin

D_sym = D * exp(-Ea / (R * (T+ 273.15) ) )

# %% [markdown]
# ##### uranium values

# %%
Ea_U = 726e3 # Activation energy in J/mol

# Pre-exponential factor
D0_U = 10**0.212 # m^2/s

# The diffusivity equation
D_U_exp = D_sym.subs({D:D0_U, Ea:Ea_U, R:8.314 }) #D0 * exp(-Ea / (R * (T+ 273.15) ) )

# Convert the sympy expression to a numpy-compatible function
D_U_fn = lambdify((T), D_U_exp, 'numpy')

D_U_exp

# %% [markdown]
# ##### Lead values

# %%
if diffusion_type.casefold() == 'fast':
    Ea_Pb = (54.6e3 *u.calorie/u.mole).to(u.joule/u.mole).m # Activation energy in J/mol
    
    # Pre-exponential factor
    D0_Pb = (1.27e-4*u.centimeter**2/u.second).to(u.meter**2/u.second).m # m^2/s

else:
    Ea_Pb = 550e3 # Activation energy in J/mol
    
    # Pre-exponential factor
    D0_Pb = 0.11 # m^2/s

# The diffusivity equation
D_Pb_exp = D_sym.subs({D:D0_Pb, Ea:Ea_Pb, R:8.314 }) # D0 * exp(-Ea / (R * (T+ 273.15) ) )

# Convert the sympy expression to a numpy-compatible function
D_Pb_fn = lambdify((T), D_Pb_exp, 'numpy')

# %%
zircon_growth_time = (model_duration - start_time) * 1e6 ### in yrs

current_ratio_U238_to_U235 = 137.818

half_life_U235 = 703.8e6  # Half-life of U-235 in years
half_life_U238 = 4.468e9  # Half-life of U-238 in years


current_U238_fraction = 0.992745
current_U235_fraction = 0.00720
current_ratio_U238_to_U235 = 137.818 ### from isoplotR


# Calculate the remaining fraction of each isotope after 1 Ga
remaining_fraction_U235 = 0.5 ** (zircon_growth_time / half_life_U235)
remaining_fraction_U238 = 0.5 ** (zircon_growth_time / half_life_U238)


# Calculate the original ratio based on remaining fractions and current ratio
# Since we know the current ratio and the remaining fractions of each isotope, we can work backwards to find the original ratio.
# The formula transforms as: original_ratio = current_ratio * (remaining_fraction_U238 / remaining_fraction_U235)
original_ratio_U238_to_U235 = current_ratio_U238_to_U235 * ( remaining_fraction_U235 / remaining_fraction_U238)

U238_N0 = original_ratio_U238_to_U235
U235_N0 = 1.
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
    Boundary0 = 1
    Boundary1 = 2
    Boundary2 = 3
    Boundary3 = 4
    Boundary4 = 5
    Boundary5 = 6
    Boundary6 = 7
    Boundary7 = 8

    
# # Points coordinates

# ### width (x) = 0.6 (60 micron), height (y) = 1. (100 micron)

# points = [(-0.3, 0.35, 0), (-0.2, 0.5, 0), (0.2, 0.5, 0), (0.3, 0.35, 0), 
#           (0.3, -0.35, 0), (0.2, -0.5, 0), (-0.2, -0.5, 0), (-0.3,-0.35, 0)]

# points_array = np.array(points)


# import gmsh

# gmsh.initialize()
# gmsh.model.add("box_with_circ")

# cellSize = csize


# # Create points
# point_ids = [gmsh.model.geo.addPoint(x, y, z, meshSize=cellSize) for x, y, z in points]

# # Create lines by connecting consecutive points and closing the loop
# line_ids = [gmsh.model.geo.addLine(point_ids[i], point_ids[(i + 1) % len(point_ids)]) for i in range(len(point_ids))]

# cl = gmsh.model.geo.addCurveLoop(line_ids)
# surface = gmsh.model.geo.addPlaneSurface([cl])

# gmsh.model.geo.synchronize()

# # Adding physical groups for lines
# for i, line_id in enumerate(line_ids):
#     boundary_tag = getattr(boundaries_2D, f"Boundary{i}")
#     gmsh.model.addPhysicalGroup(1, [line_id], tag=boundary_tag.value, name=boundary_tag.name)

# # Add physical group for the surface
# gmsh.model.addPhysicalGroup(2, [surface], 99999)
# gmsh.model.setPhysicalName(2, 99999, "Elements")

# gmsh.model.mesh.generate(2)
# gmsh.write(f'{output_path}zircon_mesh.msh')

# gmsh.finalize()

# %%
mesh = uw.discretisation.Mesh(
        mesh_file,
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
# List all available labels
num_labels = mesh.dm.getNumLabels()
# rint(f"Number of labels: {num_labels}")

# Iterate over all labels to find their names
labels = []
for i in range(num_labels):
    label_name = mesh.dm.getLabelName(i)
    labels.append(label_name)

# Filter only the boundary labels
boundary_labels = [label for label in labels if label.startswith("Boundary")]


# %%
# mesh.vtk(f"{output_path}zircon_mesh.vtk")

# import pyvista as pv
# pvmesh = pv.read(f"{output_path}zircon_mesh.vtk")

# plotter = pv.Plotter()
# plotter.add_mesh(pvmesh, show_edges=True)

# plotter.show_bounds(
#             # grid='back',
#             location='outer',
#             all_edges=True,
#             )

# plotter.view_xy()  # if mesh_2D is on the xy plane.

# plotter.save_graphic(f'{output_path}mesh_geom.pdf')
# plotter.show()

# # plotter.screenshot(f'{outputDir}{meshname}.png')  

# %%
# #### Create mesh vars


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
### swarm to store the actual point data
swarm = uw.swarm.Swarm(mesh)

Pb206_swarm = swarm.add_variable('Pb206_swarm', size=2)
Pb207_swarm = swarm.add_variable('Pb207_swarm', size=2)

U235_swarm = swarm.add_variable('U235_swarm', size=2)
U238_swarm = swarm.add_variable('U238_swarm', size=2)

time_swarm = swarm.add_variable('time', size=1)
# %%
if restart_model == False:
    swarm.add_particles_with_coordinates(Pb206.coords)
    ### Set initial U238 and U235 values
    with swarm.access(Pb206_swarm, Pb207_swarm):
        Pb206_swarm.data[...] = 0.

        Pb207_swarm.data[...] = 0.


    with swarm.access(U235_swarm, U238_swarm):
        U235_swarm.data[...] = U235_N0
        
        U238_swarm.data[...] = U238_N0


    with swarm.access(time_swarm):
        time_swarm.data[...] = 0.


else:
    restart_file = natsorted(glob(f'{timestep_dir}*swarm.fields.*xdmf'))[-1]

    swarm_restart_data = pv.read( restart_file )
    
    swarm.add_particles_with_coordinates(np.ascontiguousarray( swarm_restart_data.points[:,0:2] ))
    
    with swarm.access(Pb206_swarm, Pb207_swarm):
        Pb206_swarm.data[:,0] = swarm_restart_data['Pb206_swarm'][:,0]
        Pb206_swarm.data[:,1] = swarm_restart_data['Pb206_swarm'][:,1]

        Pb207_swarm.data[:,0] = swarm_restart_data['Pb207_swarm'][:,0]
        Pb207_swarm.data[:,1] = swarm_restart_data['Pb207_swarm'][:,1]


    with swarm.access(U235_swarm, U238_swarm):
        U235_swarm.data[:,0] = swarm_restart_data['U235_swarm'][:,0]
        U235_swarm.data[:,1] = swarm_restart_data['U235_swarm'][:,1]

        U238_swarm.data[:,0] = swarm_restart_data['U238_swarm'][:,0]
        U238_swarm.data[:,1] = swarm_restart_data['U238_swarm'][:,1]

    with swarm.access(time_swarm):
        time_swarm.data[:,0] = swarm_restart_data['time']

### update the mesh variables with the ones set on the swarm
''' may not work in parallel '''
with swarm.access(Pb206_swarm):
    with mesh.access(Pb206, Pb206_star):
        Pb206.data[:,0] = Pb206_swarm.data[:,0]
        Pb206_star.data[:,0] = Pb206_swarm.data[:,1]

with swarm.access(U238_swarm):
    with mesh.access(U238, U238_star):
        U238.data[:,0] = U238_swarm.data[:,0]
        U238_star.data[:,0] = U238_swarm.data[:,1]
with swarm.access(Pb207_swarm):
    with mesh.access(Pb207, Pb207_star):
        Pb207.data[:,0] = Pb207_swarm.data[:,0]
        Pb207_star.data[:,0] = Pb207_swarm.data[:,1]

with swarm.access(U235_swarm):
    with mesh.access(U235, U235_star):
        U235.data[:,0] = U235_swarm.data[:,0]
        U235_star.data[:,0] = U235_swarm.data[:,1]


# %%
Pb206_isotropicDiffusion = uw.systems.Poisson(mesh, u_Field=Pb206, solver_name='Pb206_diff')

U238_isotropicDiffusion = uw.systems.Poisson(mesh, u_Field=U238, solver_name='U238_diff')


Pb207_isotropicDiffusion = uw.systems.Poisson(mesh, u_Field=Pb207, solver_name='Pb207_diff')

U235_isotropicDiffusion = uw.systems.Poisson(mesh, u_Field=U235, solver_name='U235_diff')


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
total_time = nd(model_duration*u.megayear)

if restart_model == False:
    step = 0
    model_time = nd(start_time* u.megayear)
else:
    restart_step = re.search(r'swarm\.fields\.(\d+)\.xdmf', restart_file)
    step = int(restart_step.group(1))
    with swarm.access(time_swarm):
        model_time = time_swarm.data[:,0][0]



# %%
### setup Pb terms
### get current temperature from profile
Temp = temp_profile( dim(model_time, u.megayear).m ) 

D_Pb_nd = nd( (D_Pb_fn(Temp)   *u.meter**2/u.second) )

D_U_nd = nd( (D_U_fn(Temp)   *u.meter**2/u.second) )


Pb206_flux_vector = anistropic_diffusion_setup(Pb206, D_Pb_nd, D_Pb_nd)
Pb206_flux_vector_star = Pb206_flux_vector.copy()


Pb207_flux_vector = anistropic_diffusion_setup(Pb207, D_Pb_nd, D_Pb_nd)
Pb207_flux_vector_star = Pb207_flux_vector.copy()


U238_flux_vector = anistropic_diffusion_setup(U238, D_U_nd, D_U_nd)
U238_flux_vector_star = U238_flux_vector.copy()


U235_flux_vector = anistropic_diffusion_setup(U235, D_U_nd, D_U_nd)
U235_flux_vector_star = U235_flux_vector.copy()


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

def update_swarm_vars():
    ### save the swarm data too
    ### swarm data is required for restarting the model
    with mesh.access(Pb207, Pb206, U235, U238):
        Pb206_data = Pb206.data[:,0]
        Pb207_data = Pb207.data[:,0]
        U235_data = U235.data[:,0]
        U238_data = U238.data[:,0]

    with mesh.access(Pb207_star, Pb206_star, U235_star, U238_star):
        Pb206_star_data = Pb206_star.data[:,0]
        Pb207_star_data = Pb207_star.data[:,0]
        U235_star_data = U235_star.data[:,0]
        U238_star_data = U238_star.data[:,0]
        
    with swarm.access(Pb207_swarm, Pb206_swarm, U235_swarm, U238_swarm):
        Pb207_swarm.data[:,0] = Pb207_data
        Pb207_swarm.data[:,1] = Pb207_star_data

        Pb206_swarm.data[:,0] = Pb206_data
        Pb206_swarm.data[:,1] = Pb206_star_data


        U235_swarm.data[:,0] = U235_data
        U235_swarm.data[:,1] = U235_star_data

        U238_swarm.data[:,0] = U238_data
        U238_swarm.data[:,1] = U238_star_data

    with swarm.access(time_swarm):
        time_swarm.data[...] = model_time
    

def save_timestep():

    mesh.petsc_save_checkpoint(index=step, meshVars=[Pb206, Pb207, U235, U238], outputPath=timestep_dir)   

    update_swarm_vars()

    swarm.write_timestep(f'swarm', f'fields', step, [Pb206_swarm, Pb207_swarm, U235_swarm, U238_swarm, time_swarm], timestep_dir, time=model_time)

save_timestep()


# %%
while model_time < total_time:

    if step % timestep_interval == 0:
        save_timestep()

    ### get current temperature from profile
    Temp = temp_profile( dim(model_time,u.megayear).m ) 
    
    if uw.mpi.rank == 0:
        dim_time = dim(model_time, u.megayear)
        print(f'\nstep: {step}, temp: {Temp}, time: {dim_time}\n\n', flush=True)


    
    D_Pb_nd = nd( (D_Pb_fn(Temp)   *u.meter**2/u.second) )
    D_U_nd = nd( (D_U_fn(Temp)   *u.meter**2/u.second) )
    D_val = max(D_Pb_nd, D_U_nd)
    dt = min( ( (mesh.get_min_radius()**2 / D_val) ),  nd(50*u.megayear) )

    ### Finds where the timesteps can be larger due to low temps
    start_of_pulse = nd(time_arr[(temp_arr > 780)][1:][(np.diff(time_arr[temp_arr > 780]) > 100)]*u.megayear)
    end_of_pulse = nd(time_arr[temp_arr > 780][-1]*u.megayear)
    
    # if ((model_time + dt) > start_of_pulse):
    #     dt = start_of_pulse - model_time
        
    if ((model_time + dt) > start_of_pulse) & ((model_time + dt) < end_of_pulse):
        dt = min(dt, nd(5*u.megayear) )
    
    else:
        dt = dt

    if (model_time + dt) > total_time:
        dt = total_time - model_time


    ### U238 to Pb206
    diffusion_decay(U238, U238_star, Pb206, Pb206_star, U238_flux_vector_star, Pb206_flux_vector_star,
                    U238_isotropicDiffusion, Pb206_isotropicDiffusion, D_U_nd, D_Pb_nd, lambda_U238_nd, dt)

    ### U235 to Pb207
    diffusion_decay(U235, U235_star, Pb207, Pb207_star, U235_flux_vector_star, Pb207_flux_vector_star,
                    U235_isotropicDiffusion, Pb207_isotropicDiffusion, D_U_nd, D_Pb_nd, lambda_U235_nd, dt)



    step += 1
    model_time += dt



# %%
if uw.mpi.rank == 0: 
    dim_time = dim(model_time, u.megayear)
    print(f'step: {step}, time: {dim_time}\n\n', flush=True)

save_timestep()


