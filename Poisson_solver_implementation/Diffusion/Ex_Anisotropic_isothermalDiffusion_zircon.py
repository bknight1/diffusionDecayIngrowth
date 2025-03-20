# %%
import underworld3 as uw
import numpy as np
import math

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt

# %%
import os

outputPath = './output/anisotropic_isothermal_Diffusion_test/'

os.makedirs(outputPath, exist_ok=True)

# %%
csize = uw.options.getReal(name='csize', default = 0.01)

degree = uw.options.getInt(name='degree', default = 2)


### Temp to remain constant for model run
Temp = uw.options.getReal(name='tempStart', default = 800) ### C

model_duration = uw.options.getReal(name='duration', default = 500) ### Myr 


# %%
mesh_qdegree = degree

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
R = 8.314  # Gas constant in J/(mol*K)
T = symbols('T') # Temperature in Kelvin

# %%
# Activation energy, taking the mean value for the sake of this example
Ea = 726e3 # Activation energy in J/mol

# Pre-exponential factor
D0 = 10**0.212 # m^2/s

# The diffusivity equation
D_U_exp = D0 * exp(-Ea / (R * (T+ 273.15) ) )

# Convert the sympy expression to a numpy-compatible function
D_U_fn = lambdify((T), D_U_exp, 'numpy')

D_U_exp

# %%
# Activation energy, taking the mean value for the sake of this example
Ea = 550e3 # Activation energy in J/mol

# Pre-exponential factor
D0 = 0.11 # m^2/s

# The diffusivity equation
D_Pb_exp = D0 * exp(-Ea / (R * (T+ 273.15) ) )

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
mesh.vtk(f"{outputPath}zircon_mesh.vtk")

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

# plotter.screenshot(f'{outputDir}{meshname}.png')  

# %%
# #### Create mesh vars

Pb = uw.discretisation.MeshVariable("Pb", mesh, 1, degree=degree)

Pb_star = uw.discretisation.MeshVariable("Pb_star", mesh, 1, degree=degree)



# %%
with mesh.access(Pb, Pb_star):
    Pb.data[...] = 1.
    Pb_star.data[...] = 1

# %%

x_coords = np.arange(points_array[:,0].min(), points_array[:,0].max()+mesh.get_min_radius(), mesh.get_min_radius())
y_coords = np.zeros_like(x_coords)
sample_coords_x = np.column_stack([x_coords, y_coords])



Pb_init_x = uw.function.evalf(Pb.sym[0], sample_coords_x)
## BC
Pb_init_x[0], Pb_init_x[-1] = 0., 0.


# %%

y_coords = np.arange(points_array[:,1].min(), points_array[:,1].max()+mesh.get_min_radius(), mesh.get_min_radius())
x_coords = np.zeros_like(y_coords)
sample_coords_y = np.column_stack([x_coords, y_coords])



Pb_init_y = uw.function.evalf(Pb.sym[0], sample_coords_y)
## BC
Pb_init_y[0], Pb_init_y[-1] = 0., 0.

# %%
PbAnisotropicDiffusion = uw.systems.Poisson(mesh, u_Field=Pb)


# %%
### fix temp of top and bottom walls
### Pb and U boundaries set to 0
# for _solver in solvers:
for boundary in mesh.boundaries:
    PbAnisotropicDiffusion.add_dirichlet_bc(0., boundary.name)

# %%
# for _solver in solvers:
PbAnisotropicDiffusion.petsc_options['snes_rtol'] = 1e-12
PbAnisotropicDiffusion.petsc_options['snes_atol'] = 1e-6

### see the SNES output
PbAnisotropicDiffusion.petsc_options["snes_converged_reason"] = None

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
### f0 through PbAnisotropicDiffusion.f
### f1 through the flux term (above)

# %%
### setup Pb terms
D_Pb_nd = nd( (D_Pb_fn(Temp)   *u.meter**2/u.second) )


### setup flux term and history term
flux_vector = anistropic_diffusion_setup(Pb, D_Pb_nd, 0.1*D_Pb_nd)
flux_vector_star = flux_vector.copy()


# %%
def diffusion_1D(sample_points, T0, diffusivity, time_1D):
    x = sample_points
    T = T0.copy()
    k = diffusivity
    time = time_1D

    dx = sample_points[1] - sample_points[0]

    dt_dif = (dx**2 / k)

    dt = 0.5 * dt_dif


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
total_time = nd(model_duration*u.megayear)

# %%
while model_time < total_time:

    if step % 10 == 0:
        mesh.petsc_save_checkpoint(index=step, meshVars=[Pb], outputPath=outputPath)   
          
    if uw.mpi.rank == 0:
        dim_time = dim(model_time, u.megayear)
        print(f'step: {step}, time: {dim_time}\n\n')


    D_Pb_nd = nd( 2*(D_Pb_fn(Temp)   *u.meter**2/u.second) )
    dt = min( (0.5 * (mesh.get_min_radius()**2 / D_Pb_nd) ) , nd(100*u.megayear) )

    




    if (model_time + dt) > total_time:
        dt = total_time - model_time


    ### setup f0 term
    PbAnisotropicDiffusion.f = - ((Pb.sym[0] - Pb_star.sym[0]) / dt)
    ### setup flux term
    flux_vector = anistropic_diffusion_setup(Pb, 1*D_Pb_nd, 2*D_Pb_nd)

    diffusion_CM = uw.constitutive_models.DiffusionModel
    diffusion_CM.flux = flux_vector

    PbAnisotropicDiffusion.constitutive_model =  diffusion_CM
    
    theta = 0

    PbAnisotropicDiffusion.flux = theta*flux_vector + ((1-theta)*flux_vector_star)
        
    

    ### solve
    PbAnisotropicDiffusion.solve()

    ### update history terms
    flux_vector_star = flux_vector.copy()
    
    with mesh.access(Pb, Pb_star):
        Pb_star.data[...] = Pb.data[...]
    


    step += 1
    model_time += dt


    





# %%
if uw.mpi.rank == 0: 
    dim_time = dim(model_time, u.megayear)
    print(f'step: {step}, time: {dim_time}\n\n')

mesh.petsc_save_checkpoint(index=step, meshVars=[Pb], outputPath=outputPath)   

# %%
# U_final = uw.function.evalf(U238.sym[0], sample_coords)
Pb_final_x = uw.function.evalf(Pb.sym[0], sample_coords_x)
Pb_final_y = uw.function.evalf(Pb.sym[0], sample_coords_y)


# %%
Pb_1D_x = diffusion_1D(sample_coords_x[:,0], Pb_init_x, 1*D_Pb_nd, total_time)

Pb_1D_y = diffusion_1D(sample_coords_y[:,1], Pb_init_y, 2*D_Pb_nd, total_time)

# %%
plt.plot(dim(sample_coords_x[:,0], u.micrometer).m, Pb_1D_x, c='red', label='$D_{Pb}$ [x axis]')
plt.plot(dim(sample_coords_x[:,0], u.micrometer).m, Pb_final_x, ls='-.', c='k')


plt.plot(dim(sample_coords_y[:,1], u.micrometer).m, Pb_1D_y, c='salmon',label='$2D_{Pb}$ [y axis]')
plt.plot(dim(sample_coords_y[:,1], u.micrometer).m, Pb_final_y, c='k', ls='-.')
# plt.plot(dim(x_coords, u.micrometer).m, U_final, c='k', ls=':')

plt.xlabel('coord [$\mu m$]')
plt.ylabel('concentration')

plt.title(f'{Temp} $\degree $C')

plt.legend()


plt.savefig(f'{outputPath}{Temp}C-{model_duration}Myr-anisotropic_diffusion_profile.pdf')

# %%



