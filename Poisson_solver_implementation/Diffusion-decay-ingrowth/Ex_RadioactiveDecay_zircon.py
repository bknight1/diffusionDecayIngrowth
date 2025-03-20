# %%
import underworld3 as uw
import numpy as np
import math

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt

# %%
import os

outputPath = './output/RadioactiveDecay_test/'

os.makedirs(outputPath, exist_ok=True)

# %%
# Set the resolution.
res = 64

degree = 2

mesh_qdegree = degree
mesh_qdegree




# %%
# # import unit registry to make it easy to convert between units
# u = uw.scaling.units

# ### make scaling easier
# ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
# dim  = uw.scaling.dimensionalise 

# diffusive_rate    = 1e-22 * u.meter**2 /u.second
# half_life         = 52  * u.megayear
# decay_rate        = np.log(2) / half_life
# model_length      = 100 * u.micrometer ### scale the mesh radius to the garnet radius

# KL = model_length
# Kt = half_life # model_length**2 / diffusive_rate


# scaling_coefficients  = uw.scaling.get_coefficients()
# scaling_coefficients["[length]"] = KL
# scaling_coefficients["[time]"] = Kt

# scaling_coefficients

# %%
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 

### How to scale the time for diffusion and radioactive decay ???
### This works, for some reason ???
diffusive_rate    = 1e-22 * u.meter**2 /u.second


model_length      = 100 * u.micrometer ### scale the mesh radius to the zircon radius



KL = model_length
Kt = model_length**2 / diffusive_rate


scaling_coefficients  = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt

scaling_coefficients

# %%
k = 0.

# %%
import numpy as np


### time to change
start_time = 500e6  # Time period of interest in years 


# Constants
half_life_U235 = 703.8e6  # Half-life of U-235 in years
half_life_U238 = 4.468e9  # Half-life of U-238 in years

current_U238_fraction = 0.992745
current_U235_fraction = 0.00720

current_ratio_U238_to_U235 = 137.818 ### from isoplotR


# Calculate the remaining fraction of each isotope after 1 Ga
remaining_fraction_U235 = 0.5 ** (start_time / half_life_U235)
remaining_fraction_U238 = 0.5 ** (start_time / half_life_U238)


# Calculate the original ratio based on remaining fractions and current ratio
# Since we know the current ratio and the remaining fractions of each isotope, we can work backwards to find the original ratio.
# The formula transforms as: original_ratio = current_ratio * (remaining_fraction_U238 / remaining_fraction_U235)
original_ratio_U238_to_U235 = current_ratio_U238_to_U235 * ( remaining_fraction_U235 / remaining_fraction_U238)

U238_N0 = original_ratio_U238_to_U235
U235_N0 = 1.



# %%
current_ratio_U238_to_U235

# %%
half_life_U235 = 703.8e6 * u.year # Half-life of U-235 in years
half_life_U238 = 4.468e9 * u.year # Half-life of U-238 in years

# Decay constant of U238
lambda_U238 = np.log(2) / half_life_U238
lambda_U235 = np.log(2) / half_life_U235


lambda_U238_nd = nd(lambda_U238)

lambda_U235_nd = nd(lambda_U235)

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

cellSize = 0.01


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
gmsh.write(f'{outputPath}/zircon_mesh.msh')

gmsh.finalize()

# %%
mesh = uw.discretisation.Mesh(
        f'{outputPath}/zircon_mesh.msh',
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

# Create mesh vars
U238 = uw.discretisation.MeshVariable("U238", mesh, 1, degree=degree)
U235 = uw.discretisation.MeshVariable("U235", mesh, 1, degree=degree)

Pb206 = uw.discretisation.MeshVariable("P206", mesh, 1, degree=degree)
Pb207 = uw.discretisation.MeshVariable("P207", mesh, 1, degree=degree)



# %%


# %%
U238diffusion = uw.systems.Poisson(mesh, U238)
U235diffusion = uw.systems.Poisson(mesh, U235)

solvers = [U238diffusion, U235diffusion]

# %%
# ##### Set up solver parameters

for _solver in solvers:
    _solver.constitutive_model = uw.constitutive_models.DiffusionModel
    _solver.constitutive_model.Parameters.diffusivity = k

    _solver.constitutive_model.Parameters.diffusivity

# %%
### fix temp of top and bottom walls
### Pb and U boundaries set to 0
for _solver in solvers:
    for boundary in mesh.boundaries:
        _solver.add_dirichlet_bc(0., boundary.name)


with mesh.access(U238):
    U238.data[...] = U238_N0

with mesh.access(U235):
    U235.data[...] = U235_N0




# %%
for _solver in solvers:
    _solver.petsc_options['snes_rtol'] = 1e-12
    _solver.petsc_options['snes_atol'] = 1e-6

    ### see the SNES output
    _solver.petsc_options["snes_converged_reason"] = None

# %%
# Udiffusion.f = (-1*lambda_U_nd*U.sym[0])*(dt/nd(half_life1))

# Pbdiffusion.f = (-1*lambda_Pb_nd*Pb.sym[0])*(dt/nd(half_life2))

# %%
step = 0
model_time = 0.



nsteps = 10

dt = nd(start_time*u.year) / nsteps

# %%
start_time

# %%
U238_values, U235_values, Pb207_values, Pb206_values = [U238_N0], [U235_N0], [0.], [0.]

time_values = [0.]
while step < nsteps:

    if step % 1 == 0:
        # mesh.petsc_save_checkpoint(index=step, meshVars=[U], outputPath=outputPath)     
        if uw.mpi.rank == 0:
            dim_time = dim(model_time, u.megayear)
            print(f'step: {step}, time: {dim_time}\n\n')

    # Pb_sample = uw.function.evalf(Pb.sym[0], np.array([[0.,0.]]))

    # Pb_values.append(Pb_sample)

    ### solve
    # U238diffusion.solve()
    # U235diffusion.solve()

    U238_lost_fn  =  U238.sym[0] - (U238.sym[0] * np.exp(-lambda_U238_nd*dt))
    U235_lost_fn  =  U235.sym[0] - (U235.sym[0] * np.exp(-lambda_U235_nd*dt))

    U238_lost_data = uw.function.evalf(U238_lost_fn, U238.coords)
    U235_lost_data = uw.function.evalf(U235_lost_fn, U235.coords)

    ### Update values and copy to history term
    with mesh.access(U238):
        U238.data[:,0] -= U238_lost_data
        # U238diffusion.u_Star.data[:,0] = np.copy(U238.data[:,0])

    with mesh.access(Pb206):
        Pb206.data[:,0] += U238_lost_data

    ### Update values and copy to history term
    with mesh.access(U235):
        U235.data[:,0] -= U235_lost_data
        # U235diffusion.u_Star.data[:,0] = np.copy(U235.data[:,0])

    with mesh.access(Pb207):
        Pb207.data[:,0] += U235_lost_data





    step += 1
    model_time += dt

    U238_sample = uw.function.evalf(U238.sym[0], np.array([[0.,0.]]))
    U235_sample = uw.function.evalf(U235.sym[0], np.array([[0.,0.]]))
    Pb207_sample = uw.function.evalf(Pb207.sym[0], np.array([[0.,0.]]))
    Pb206_sample = uw.function.evalf(Pb206.sym[0], np.array([[0.,0.]]))

    U238_values.append(U238_sample)
    U235_values.append(U235_sample)
    Pb207_values.append(Pb207_sample)
    Pb206_values.append(Pb206_values)
    time_values.append(model_time)

    





# %%
if uw.mpi.rank == 0:
    dim_time = dim(model_time, u.megayear)
    print(f'step: {step}, time: {dim_time}\n\n')

# %%

# x_coords = np.linspace(points_array[:,0].min()+0.2, points_array[:,0].max()-0.2, 101)
# y_coords = np.zeros_like(x_coords)
# sample_coords = np.column_stack([x_coords, y_coords])


# U_sample = uw.function.evalf(U.sym[0], sample_coords)



# %%
# lambda_U

# %%

remaining_U238 = U238_N0 * np.exp(-lambda_U238_nd*np.array(time_values))
remaining_U235 = U235_N0 * np.exp(-lambda_U235_nd*np.array(time_values))

# %%
remaining_U238

# remaining_Pb

# %%
plt.plot((time_values), U238_values)
plt.plot(time_values, remaining_U238, ls=':', label='analytical solution')

# plt.plot(nd(t_arr*u.megayear), remaining_Pb, ls=':')

# %%
import numpy as np
import matplotlib.pyplot as plt

def concordia_curve(t, lambda_235=9.8485e-10, lambda_238=1.55125e-10):
    # Calculate the exponential growth for both decay systems
    e_lambda_235t = np.exp(lambda_235 * t)
    e_lambda_238t = np.exp(lambda_238 * t)
    
    # Calculate ratios
    Pb206_U238 = e_lambda_238t - 1
    Pb207_U235 = e_lambda_235t - 1
    Pb207_Pb206 = (Pb207_U235 / Pb206_U238) * (1 / current_ratio_U238_to_U235)  # Natural ratio of U235/U238

    return Pb207_Pb206, Pb206_U238

# Time range in years
time = np.linspace(10e6, 1200e6, 1000)  # 0 to 1000 Ma

time1 = np.linspace(500e6, 1000e6, 6)

time2 = np.linspace(550e6, 1050e6, 6)



# Generate data for the Concordia curve
pb207_pb206, pb206_u238 = concordia_curve(time)

pb207_pb206_1, pb206_u238_1 = concordia_curve(time1)

pb207_pb206_2, pb206_u238_2 = concordia_curve(time2)



# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(1/pb206_u238_1, pb207_pb206_1, c='green', )
plt.plot(1/pb206_u238, pb207_pb206, label='Concordia Curve', c='k')
for i in range(len(pb206_u238_1)):
    plt.text( (1/pb206_u238_1[i])+0.1, pb207_pb206_1[i], f'{round(time1[i]/1e6)} Ma')
plt.xlabel('$^{238}U/^{206}Pb$')
plt.ylabel('$^{207}Pb/^{206}Pb$')
plt.title('U-Pb Concordia')
plt.xlim(5.5, 12.5)
plt.ylim(0.055, 0.075)
# plt.legend()
plt.grid(True)

# time2 = np.linspace(550e6, 1050e6, 6)
# pb207_pb206_2, pb206_u238_2 = concordia_curve(time2)
# plt.scatter(1/pb206_u238_2, pb207_pb206_2, marker='x', c='r', s=100 )

plt.scatter(1/pb206_u238_2, pb207_pb206_2, marker='x', c='r', s=100)

# plt.savefig('U-Pb_isotope_concordia_plot.pdf')

plt.show()



# %%



