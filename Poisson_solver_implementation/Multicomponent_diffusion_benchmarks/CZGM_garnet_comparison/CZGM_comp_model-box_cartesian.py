# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: uw
#     language: python
#     name: python3
# ---

# %%
import underworld3 as uw

import pandas as pd
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
# import GarnetDiffusion

import math
import sympy as sp

import os

# %%
dt_fac = 10.

outputDir = f'./output/CZGM_comp_model_box_cartesian_{dt_fac}dt/'

restart = False


if uw.mpi.rank == 0:
    os.makedirs(outputDir, exist_ok=True)

    # ### copy file to directory
    # for file in glob.glob(fileName):
    #     shutil.copy(file, outputDir)

# %% [markdown]
# #### extract the (P)Tt path

# %%

CZGM_EM_data = pd.read_csv('./CZGM_data/combined_array.csv', header=None)
CZGM_PTt_path = pd.read_csv('./CZGM_data/PTt_path.csv', header=None)

# %%
plt.plot(CZGM_PTt_path.iloc[:,1]-273.15, CZGM_PTt_path.iloc[:,0]/1e4)

plt.xlabel('T [$\degree$C]')
plt.ylabel('P [GPa]')
plt.grid(ls= ':', alpha=0.5)

# %%
#### create a 1D interp of Tt path to accurately represent the path in the modelling
Tt_path_interp = interpolate.interp1d(CZGM_PTt_path.iloc[:,2], CZGM_PTt_path.iloc[:,1], fill_value="extrapolate")

Pt_path_interp = interpolate.interp1d(CZGM_PTt_path.iloc[:,2], CZGM_PTt_path.iloc[:,0], fill_value="extrapolate")

# %% [markdown]
# ### Extract the radial garnet evolution

# %%
Rr, tr, Pr, Tr, Fer, Mgr, Mnr, Car = (CZGM_EM_data.iloc[:, i].values for i in range(8))

# %%
plt.plot(Rr, Fer, label='Fe', c='deeppink')

plt.plot(Rr, Mgr, label='Mg', c='red')

plt.plot(Rr, Mnr, label='Mn', c='blue')

plt.plot(Rr, Car, label='Ca', c='green')


# %% [markdown]
# ### Scaling the model

# %%


# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 


diffusive_rate    = 1e-20 * u.meter**2 /u.second


model_length      = Rr.max() * u.micrometer ### scale the mesh radius to the final garnet size



KL = model_length
Kt = model_length**2 / diffusive_rate


scaling_coefficients  = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt

scaling_coefficients

# %% [markdown]
# ### Extract radial growth of garnet

# %%
Rr_nd = nd(Rr*u.micrometer) ### For the mesh generation
Rr_nd

# %%
### Extract the growth part of the evolution
radial_growth      = (Rr[tr>tr.min()]) ### in micrometer
time_evolution     = tr[tr>tr.min()] ### Myr
pressure_evolution = Pr[tr>tr.min()]
# pressure_evolution = (P_evolution*u.kilobar).m*1e3 ### convert to bar from kbar
temp_evolution     = Tr[tr>tr.min()] ### in K
Mg_evolution       = Mgr[tr>tr.min()]
Fe_evolution       = Fer[tr>tr.min()]
Ca_evolution       = Car[tr>tr.min()]
Mn_evolution       = Mnr[tr>tr.min()] # 0.05*np.ones_like(P_evolution) #

# %%
degree = uw.options.getInt(name='degree', default = 1)

# %% [markdown]
# ### Create the mesh

# %%
from enum import Enum

boundary_labels = ['Centre']
boundary_tags = [100]


if uw.mpi.rank == 0:
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", True)
    gmsh.model.add("Annulus_garnet_mesh")

    def generate_curve(radius, cellsize, p0, label):
        ### creates a curve for each radial increase in the garnet

        p1 = gmsh.model.geo.add_point(
            radius, 0.0, 0.0, meshSize=cellsize
        )
        p2 = gmsh.model.geo.add_point(
            -radius, 0.0, 0.0, meshSize=cellsize
        )

        c0 = gmsh.model.geo.add_circle_arc(p1, p0, p2)
        c1 = gmsh.model.geo.add_circle_arc(p2, p0, p1)


        cl = gmsh.model.geo.add_curve_loop([c0, c1], tag=label)


        return [c0, c1], cl
    


    loops = []

    internal_curves = []
    cls = []

    cellsize = np.diff(Rr_nd)[0]

    p0 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=cellsize)

    i = 101
    for radius in Rr_nd[1:]:
        internal_curve, cl = generate_curve(radius, cellsize, p0, i )
        cls.append( cl) 
        internal_curves.append( internal_curve )
        i += 1
        
    
    loops = [cl] + loops

    s = gmsh.model.geo.add_plane_surface(loops)

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.embed(0, [p0], 2, s)

    gmsh.model.geo.synchronize()

    # ### embend the curves for the garnet growth
    for curve in internal_curves:
        gmsh.model.mesh.embed(1, curve, 2, s)

    gmsh.model.geo.synchronize()

    
    gmsh.model.addPhysicalGroup(
            0, [p0], tag=boundary_tags[0], name=boundary_labels[0]
        )

    # gmsh.model.addPhysicalGroup(
    #     1,
    #     internal_curves[-1],
    #     102,
    #     name='Upper',
    # )

    shell = 0
    start_shell = Rr.shape[0] - radial_growth.shape[0] - 1  ### shell at which the growth over time starts at
    
    for curve in internal_curves[start_shell:]:
        boundary_label = f'shell_{shell}'
        boundary_labels.append(boundary_label)
        boundary_tags.append( cls[shell] )
        gmsh.model.addPhysicalGroup(
        1,
        curve,
        cls[shell],
        name=boundary_label,
    )
        shell += 1


    gmsh.model.addPhysicalGroup(2, [s], 666666, "Elements")
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write(f'{outputDir}/garnet_mesh.msh')
    gmsh.finalize()


# %%
# Create a dictionary from the two lists
members = dict(zip(boundary_labels, boundary_tags))

# Create a new Enum
boundaries = Enum("boundaries", members)

# %%
# mesh = uw.discretisation.Mesh(
#     f'{outputDir}/garnet_mesh.msh',
#         degree=1,
#         qdegree=degree,
#         boundaries=boundaries,
#         boundary_normals=None,
#         coordinate_system_type=uw.coordinates.CoordinateSystemType.CARTESIAN,
#         useMultipleTags=True,
#         useRegions=True,
#         markVertices=True,
#         refinement=None,
#         refinement_callback=None,
#     )

# %%
xmin, xmax = 0, nd(Rr.max()*u.micrometer)
ymin, ymax = 0, xmax/10

xres = Rr.shape[0] - 1 
yres = int ( round( (ymax/xmax)*xres ) )

mesh = uw.meshing.StructuredQuadBox( minCoords=(xmin, ymin), maxCoords=(xmax, ymax), elementRes=(xres,yres) )

# %%
# mesh.view()

# %%
# if uw.mpi.size == 1:
#     import pyvista as pv

#     mesh.vtk(f"{outputDir}Garnet_mesh.vtk")


#     pvmesh = pv.read(f"{outputDir}Garnet_mesh.vtk")

#     plotter = pv.Plotter()
#     plotter.add_mesh(pvmesh, show_edges=True)
#     plotter.view_xy()  # if mesh_2D is on the xy plane.
#     plotter.show()

# %% [markdown]
# #### Setup mesh variables for diffusion

# %%
#### add mesh vars for each element
u_continuous = True
Fe = uw.discretisation.MeshVariable("Fe", mesh, 1, degree=degree, continuous=u_continuous)

Fe_star = uw.discretisation.MeshVariable("Fe_star", mesh, 1, degree=degree, continuous=u_continuous)


Mg = uw.discretisation.MeshVariable("Mg", mesh, 1, degree=degree, continuous=u_continuous)
Mg_star = uw.discretisation.MeshVariable("Mg_star", mesh, 1, degree=degree, continuous=u_continuous)

# Create Mn variables
Mn = uw.discretisation.MeshVariable("Mn", mesh, 1, degree=degree, continuous=u_continuous)
Mn_star = uw.discretisation.MeshVariable("Mn_star", mesh, 1, degree=degree, continuous=u_continuous)


Ca = uw.discretisation.MeshVariable("Ca", mesh, 1, degree=degree, continuous=u_continuous)
Ca_star = uw.discretisation.MeshVariable("Ca_star", mesh, 1, degree=degree, continuous=u_continuous)

# %% [markdown]
# ##### Create profile coords to sample each end member

# %%
### for saving profiles
profile_coords = np.zeros((Rr_nd.shape[0],2))
#### profile along garnet from core to rim along x axis, matching the mesh nodal points
profile_coords[:,0] = np.linspace(Rr_nd.min(), Rr_nd.max(), Rr_nd.shape[0])
profile_coords[:,1] = 0.

# %% [markdown]
# ### Create the solver for each element

# %%
# Create diffusion system for Fe

diff_Fe = uw.systems.Poisson(mesh, u_Field=Fe)

# Create diffusion system for Mg

diff_Mg = uw.systems.Poisson(mesh, u_Field=Mg)

# Create diffusion system for Mn

diff_Mn = uw.systems.Poisson(mesh, u_Field=Mn)

# %% [markdown]
# ### Set up initial concentrations

# %%
with mesh.access(Fe, Mg, Mn, Ca, Fe_star, Mg_star, Mn_star, Ca_star):
    ### Set everything to the initial value of the Garnet
    Fe.data[:,0] = Fer[0]
    Fe_star.data[:,0] = Fer[0]
    # Fe_star.data[:,0] = Fer[0]

    Mg.data[:,0] = Mgr[0]
    Mg_star.data[:,0] = Mgr[0]
    # Mg_star.data[:,0] = Mgr[0]

    Mn.data[:,0] = Mnr[0]
    Mn_star.data[:,0] = Mgr[0]
    # Mn_star.data[:,0] = Mnr[0]

    # Ca.data[:,0] = Car[tr==tr.min()][-1]
    Ca.data[:,0] = 1 - Fe.data[:,0] - Mg.data[:,0] - Mn.data[:,0]
    Ca_star.data[:,0] = 1 - Fe.data[:,0] - Mg.data[:,0] - Mn.data[:,0]

# %%
for _solver_ in [diff_Fe, diff_Mg, diff_Mn]:
    _solver_.constitutive_model = uw.constitutive_models.DiffusionModel
    # _solver_.constitutive_model.Parameters.diffusivity = 1.0
    _solver_.petsc_options['snes_rtol'] = 1e-8
    _solver_.petsc_options['snes_atol'] = 1e-8
    _solver_.petsc_options["snes_converged_reason"] = None

# %% [markdown]
# #### Function to create the diffusion matrix

# %%
import sympy as sp


def create_diffusion_matrix(D_star, X):
    """
    Create a symbolic diffusion matrix for N components based on the given formula:
    
    D_ij = D_i * delta_ij - [ (D_i * X_i) / (sum_k D_k * X_k) ] * (D_j - D_Ca)

    Parameters:
    - D_star: List of symbolic diffusion coefficients D*_i for each component.
    - X: List of symbolic mole fractions X_i for each component.

    Returns:
    - D: SymPy matrix representing the diffusion matrix.
    """
    N = len(D_star)  # Number of components

    # Create symbolic delta_ij (Kronecker delta)
    delta = lambda i, j: 1 if i == j else 0

    # Create symbolic matrix D_ij
    D = sp.Matrix.zeros(N, N)

    # Calculate the normalization term (denominator)
    denom = sum(D_star[k] * X[k] for k in range(N))

    # Fill in the diffusion matrix
    for i in range(N):
        for j in range(N):
            # Apply the given formula
            term_1 = D_star[i] * delta(i, j)
            term_2 = (D_star[i] * X[i] ) / denom * ( (D_star[j] - D_star[-1]) )  # Assuming last component is Ca

            D[i, j] = term_1 - term_2

    return D


# %%
def update_kappa(temp, pressure):
    ### determine the diffusion rate at the current T and P

    ### pressure in bars
    ### temp in Kelvin

    ### k in cm2/s

    # Define the symbols
    # R = 8.314  # Gas constant in J/(K mol)
    # R = 1.9872 ### in cal/(K mol)
    
    ### Symbolic Arrenhius eq.
    D, Ea, Ev, R, T, P = sp.symbols('D Ea Ev R T P') # Temperature in Kelvin
    
    D_sym = D * sp.exp(-(Ea + (P-1)*Ev/41.84) / (R * T ) )



    ### From Chakraborty and Ganguly  1992 (with a correction)
    DDMn=5.15e-4 ### in cm^2/s
    DDMg=1.11e-3
    DDFe=6.36e-4

    # temp +=273.15 ### in K
    # pressure *=1e3 ### in bar

    EaFe = 65824
    EaMg = 67997
    EaMn = 60569

    EvFe = 5.63
    EvMg = 5.27
    EvMn = 6.04

    D_Fe = nd( float(D_sym.subs({D:DDFe, Ea:EaFe, Ev:EvFe, R:1.9872, T:temp, P:pressure })) *u.centimeter**2/u.second)
    D_Mg = nd( float(D_sym.subs({D:DDMg, Ea:EaMg, Ev:EvMg, R:1.9872, T:temp, P:pressure })) *u.centimeter**2/u.second)
    D_Mn = nd( float(D_sym.subs({D:DDMn, Ea:EaMn, Ev:EvMn, R:1.9872, T:temp, P:pressure })) *u.centimeter**2/u.second)
    D_Ca = D_Mn*0.05
    
    
    # Create the diffusion matrix
    D_star = [D_Fe, D_Mg, D_Mn, D_Ca]
    xi = [Fe.sym[0], Mg.sym[0], Mn.sym[0], Ca.sym[0]]
    
    D_matrix = create_diffusion_matrix(D_star, xi)

    D_max = np.float64( np.max(D_star) )


    return D_matrix, D_max


# %% [markdown]
# #### Functions to setup solvers

# %%
def update_BC(solver, element_conc, boundary):
    ### annulus, only set BC on the outside boundary
    solver.add_dirichlet_bc(element_conc, boundary)

def update_domain(step, shell_radius):
    ### add the new 'shell' to the garnet
    r = mesh.X[0] #sp.sqrt( mesh.X[0]**2 + mesh.X[1]**2 )
    r_vals = uw.function.evaluate(r, Fe.coords)
    with mesh.access(Fe, Mg, Mn, Ca):
        update_area = (r_vals > (nd(shell_radius*u.micrometer) - (0.2*mesh.get_min_radius())  ) )  
        ### everything outside the current radius is equal to BC
        Fe.data[update_area] = Fe_evolution[step]
        Mg.data[update_area] = Mg_evolution[step]
        Mn.data[update_area] = Mn_evolution[step]
        Ca.data[update_area] = Ca_evolution[step]

    ### reset BC and update
    diff_Fe.essential_bcs = []
    diff_Mg.essential_bcs = []
    diff_Mn.essential_bcs = []

    update_BC(diff_Fe, Fe_evolution[step], 'Right')
    update_BC(diff_Mg, Mg_evolution[step], 'Right')
    update_BC(diff_Mn, Mn_evolution[step], 'Right')  

    # ### loop over boundaries and update all those from the current shell up
    # for boundary in mesh.boundaries:
    #     try:
    #         boundary_name = boundary.name
    #         shell = int( boundary_name.split("_")[1] )
    #         if shell >= step:
    #             # print(f'updating boundary: {boundary_name}')
    #             update_BC(diff_Fe, Fe_evolution[step], boundary_name)
    #             update_BC(diff_Mg, Mg_evolution[step], boundary_name)
    #             update_BC(diff_Mn, Mn_evolution[step], boundary_name)   
    #     except:
    #         pass

    # print(f'step = {step}, nobc = {len(diff_Fe.essential_bcs)}' )


# def gradient_calc(C_sym):
#     """Calculates the gradient of concentration C in the x and y component"""
#     # Assume x, y as spatial coordinates
#     if mesh.dim == 3:
#         x, y, z = mesh.X
#     else:
#         x, y = mesh.X

#     # Compute the gradients of each concentration
#     grad_C_x = sp.Matrix([sp.diff(C_sym, x)])
#     grad_C_y = sp.Matrix([sp.diff(C_sym, y)])

#     gradients = sp.Matrix([grad_C_x, grad_C_y])
    
#     if mesh.dim == 3:
#         grad_C_z = sp.Matrix([sp.diff(C_sym, z)])

#         gradients = sp.Matrix([[grad_C_x], [grad_C_y], [grad_C_z]])

#     return gradients

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
    
    # if mesh.dim == 3:
    #     grad_C_z = sp.Matrix([sp.diff(C_sym, z)])

    #     gradients = sp.Matrix([[grad_C_x], [grad_C_y], [grad_C_z]])

    return gradients


# %%
def update_history_terms():
    with mesh.access(Fe, Fe_star):
        Fe_star.data[:,0] = np.copy(Fe.data[:,0])

    with mesh.access(Mg, Mg_star):
        Mg_star.data[:,0] = np.copy(Mg.data[:,0])
        
    with mesh.access(Mn, Mn_star):
        Mn_star.data[:,0] = np.copy(Mn.data[:,0])

    with mesh.access(Ca, Fe, Mg, Mn, Ca_star):
        # Ca.data[:,0] = 1 - Fe.data[:,0] - Mg.data[:,0] - Mn.data[:,0] 
        Ca_star.data[:,0] = np.copy(Ca.data[:,0])


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
# ### Symbolic representation of $\nabla C$

# %%
nabla_Fe, nabla_Mg, nabla_Mn = sp.symbols(r'\nabla_Fe \nabla_Mg \nabla_Mn')


grad_matrix = sp.Matrix([[nabla_Fe], [nabla_Mg], [nabla_Mn]])

# %%
plt.plot(CZGM_PTt_path.iloc[:,2], CZGM_PTt_path.iloc[:,1], )
plt.scatter(CZGM_PTt_path.iloc[:,2], CZGM_PTt_path.iloc[:,1], )
# plt.plot([dim(model_time, u.megayear).m, dim(model_time, u.megayear).m], [0, 1000])
plt.ylim(700, 900)
plt.grid()

# %% [markdown]
# #### Diffusion along the prograde path

# %%
step = 0
model_time = nd(time_evolution[0] * u.megayear)

# %%
#### Diffusion while growing
### from the first timestep of the shell to the lat timestep
while model_time < nd(time_evolution[-1]*u.megayear):

    if uw.mpi.rank == 0:
        print(f"Step: {str(step).rjust(3)}, time: {dim(model_time, u.megayear)}\n")

    ### Grow the garnet in the domain
    update_domain(step, radial_growth[step])

    temperature_for_D = Tt_path_interp(dim(model_time, u.megayear).m) # temp_evolution[step]
    pressure_for_D =  Pt_path_interp(dim(model_time, u.megayear).m) # pressure_evolution[step]
    ### determine the diffusion matrix from current temp
    D_term, D_max = update_kappa( temperature_for_D, pressure_for_D)

    ### setup flux (J) term from D matrix and nabla C
    flux_term_sym = D_term[:3,:3] * grad_matrix
    ### sub in the actual nabla C terms
    flux_terms = flux_term_sym.subs( {nabla_Fe:gradient_calc(Fe.sym[0]), nabla_Mg:gradient_calc(Mg.sym[0]), nabla_Mn:gradient_calc(Mn.sym[0]) } )

    ### extract flux terms
    Fe_flux_term = flux_terms[0]
    Mg_flux_term = flux_terms[1]
    Mn_flux_term = flux_terms[2]

    Fe_flux_star = Fe_flux_term.copy()
    Mg_flux_star = Mg_flux_term.copy()
    Mn_flux_star = Mn_flux_term.copy()

    ### update history terms
    update_history_terms()

    #### save the end member values
    mesh.petsc_save_checkpoint(step, meshVars=[Fe, Mg, Mn, Ca], outputPath=outputDir)


    Fe_profile_1 = uw.function.evaluate(Fe.sym[0], profile_coords)
    Mg_profile_1 = uw.function.evaluate(Mg.sym[0], profile_coords)
    Mn_profile_1 = uw.function.evaluate(Mn.sym[0], profile_coords)
    Ca_profile_1 = uw.function.evaluate(Ca.sym[0], profile_coords)
    
    plt.plot(profile_coords[:,0]*Rr.max(), Fe_profile_1, c='red')
    plt.plot(profile_coords[:,0]*Rr.max(), Mg_profile_1, c='green')
    plt.plot(profile_coords[:,0]*Rr.max(), Mn_profile_1, c='blue')
    plt.plot(profile_coords[:,0]*Rr.max(), Ca_profile_1, c='gold')
    
    plt.plot(Rr, Fer, label='Fe', ls=':', c='red')
    plt.plot(Rr, Mgr, label='Mg', ls=':', c='green')
    plt.plot(Rr, Mnr, label='Mn', ls=':', c='blue')
    plt.plot(Rr, Car, label='Ca', ls=':', c='gold')

    plt.title(f'time = {round(dim(model_time, u.megayear).m, 2)} Myr')

    plt.xlabel('r [$\mu m$]')
    plt.ylabel('F')

    
    plt.grid()
    plt.legend()

    plt.savefig(f'{outputDir}step.{step}.diffusion_profiles.pdf')

    plt.close()


    internal_step = 0
    ### keep looping over until we reach the next time iteration
    while model_time < nd(time_evolution[step]*u.megayear):
        
        temperature_for_D = Tt_path_interp(dim(model_time, u.megayear).m) # temp_evolution[step]
        pressure_for_D =  Pt_path_interp(dim(model_time, u.megayear).m) # pressure_evolution[step]
        ### determine the diffusion matrix from current temp
        D_term, D_max = update_kappa( temperature_for_D, pressure_for_D)
        
        dt = dt_fac * (mesh.get_min_radius()**2 / D_max)
        print(f'\n\ninternal step = {internal_step}, model time = {dim(model_time, u.megayear)}, temperature = {Tt_path_interp(dim(model_time, u.megayear).m)}\n\n')
        ### if model time + dt is less than the next time we want, just use dt
        if model_time+dt < nd(time_evolution[step]*u.megayear):
            dt = dt
        else:
            ### If it exceeds the time we're solving over we determine dt to the difference 
            dt = (nd(time_evolution[step]*u.megayear) - model_time)

        with mesh.access(Ca, Fe, Mg, Mn):
            Ca.data[:,0] = 1 - Fe.data[:,0] - Mg.data[:,0] - Mn.data[:,0] 

        flux_term_sym = D_term[:3,:3] * grad_matrix
        flux_terms = flux_term_sym.subs( {nabla_Fe:gradient_calc(Fe.sym[0]), nabla_Mg:gradient_calc(Mg.sym[0]), nabla_Mn:gradient_calc(Mn.sym[0]) } )
    
        Fe_flux_term = flux_terms[0]
        Mg_flux_term = flux_terms[1]
        Mn_flux_term = flux_terms[2]

        
        
        ### solve the equation
        solve_diff_eq(diff_Fe, Fe, Fe_star, Fe_flux_term, Fe_flux_star, dt)
        solve_diff_eq(diff_Mg, Mg, Mg_star, Mg_flux_term, Mg_flux_star, dt)
        solve_diff_eq(diff_Mn, Mn, Mn_star, Mn_flux_term, Mn_flux_star, dt)

        ### save history terms
        Fe_flux_star = Fe_flux_term.copy()
        Mg_flux_star = Mg_flux_term.copy()
        Mn_flux_star = Mn_flux_term.copy()


        update_history_terms()


        model_time += dt
        internal_step += 1


    ### move up a step after reaching the time where the next shell grows
    step += 1

# %%
Fe_profile_EndGrowth = uw.function.evaluate(Fe.sym[0], profile_coords)
Mg_profile_EndGrowth = uw.function.evaluate(Mg.sym[0], profile_coords)
Mn_profile_EndGrowth = uw.function.evaluate(Mn.sym[0], profile_coords)
Ca_profile_EndGrowth = uw.function.evaluate(Ca.sym[0], profile_coords)

plt.plot(profile_coords[:,0]*Rr.max(), Fe_profile_EndGrowth, c='red', lw=0.5)
plt.plot(profile_coords[:,0]*Rr.max(), Mg_profile_EndGrowth, c='green', lw=0.5)
plt.plot(profile_coords[:,0]*Rr.max(), Mn_profile_EndGrowth, c='blue', lw=0.5)
plt.plot(profile_coords[:,0]*Rr.max(), Ca_profile_EndGrowth, c='gold', lw=0.5)


plt.plot(Rr, Fer, label='Fe', ls=':', c='red')
plt.plot(Rr, Mgr, label='Mg', ls=':', c='green')
plt.plot(Rr, Mnr, label='Mn', ls=':', c='blue')
plt.plot(Rr, Car, label='Ca', ls=':', c='gold')

plt.grid()
plt.legend()

plt.show()

# %%
plt.plot(CZGM_PTt_path.iloc[:,2], CZGM_PTt_path.iloc[:,1], )
plt.plot([dim(model_time, u.megayear).m, dim(model_time, u.megayear).m], [0, 1000])
plt.ylim(600, 900)
plt.grid()

# %% [markdown]
# #### Diffusion along the post-peak garnet growth / retrograde path

# %%
### reset BC and update
diff_Fe.essential_bcs = []
diff_Mg.essential_bcs = []
diff_Mn.essential_bcs = []

update_BC(diff_Fe, Fe_evolution[-1], 'Right')
update_BC(diff_Mg, Mg_evolution[-1], 'Right')
update_BC(diff_Mn, Mn_evolution[-1], 'Right')  

# %%

P_retro, T_retro, t_retro =  (CZGM_PTt_path.iloc[:, i][CZGM_PTt_path.iloc[:,2] > dim(model_time, u.megayear).m].values for i in range(3))

retro_step = 0

# %%
#### Diffusion while growing
### from the first timestep of the shell to the lat timestep
while model_time < nd(t_retro[-1]*u.megayear):

    if uw.mpi.rank == 0:
        print(f"Step: {str(step).rjust(3)}, retro_step: {str(retro_step).rjust(3)}, time: {dim(model_time, u.megayear)}\n")

    ### Grow the garnet in the domain
    # update_domain(step, radial_growth[step])

    temperature_for_D = Tt_path_interp(dim(model_time, u.megayear).m) # T_retro[retro_step]
    pressure_for_D =  Pt_path_interp(dim(model_time, u.megayear).m) # P_retro[retro_step]
    ### determine the diffusion matrix from current temp
    D_term, D_max = update_kappa( temperature_for_D, pressure_for_D)

    flux_term_sym = D_term[:3,:3] * grad_matrix
    flux_terms = flux_term_sym.subs( {nabla_Fe:gradient_calc(Fe.sym[0]), nabla_Mg:gradient_calc(Mg.sym[0]), nabla_Mn:gradient_calc(Mn.sym[0]) } )


    Fe_flux_term = flux_terms[0]
    Mg_flux_term = flux_terms[1]
    Mn_flux_term = flux_terms[2]

    Fe_flux_star = Fe_flux_term.copy()
    Mg_flux_star = Mg_flux_term.copy()
    Mn_flux_star = Mn_flux_term.copy()


    update_history_terms()

        
    mesh.petsc_save_checkpoint(step, meshVars=[Fe, Mg, Mn, Ca], outputPath=outputDir)


    Fe_profile_1 = uw.function.evaluate(Fe.sym[0], profile_coords)
    Mg_profile_1 = uw.function.evaluate(Mg.sym[0], profile_coords)
    Mn_profile_1 = uw.function.evaluate(Mn.sym[0], profile_coords)
    Ca_profile_1 = uw.function.evaluate(Ca.sym[0], profile_coords)
    
    plt.plot(profile_coords[:,0]*Rr.max(), Fe_profile_1, c='blue')
    plt.plot(profile_coords[:,0]*Rr.max(), Mg_profile_1, c='red')
    plt.plot(profile_coords[:,0]*Rr.max(), Mn_profile_1, c='green')
    plt.plot(profile_coords[:,0]*Rr.max(), Ca_profile_1, c='orange')
    
    plt.plot(Rr, Fer, label='Fe', ls=':', c='blue')
    plt.plot(Rr, Mgr, label='Mg', ls=':', c='red')
    plt.plot(Rr, Mnr, label='Mn', ls=':', c='green')
    plt.plot(Rr, Car, label='Ca', ls=':', c='orange')

    plt.title(f'time = {round(dim(model_time, u.megayear).m, 2)} Myr')

    plt.xlabel('r [$\mu m$]')
    plt.ylabel('F')

    
    plt.grid()
    plt.legend()

    plt.savefig(f'{outputDir}step.{step}.diffusion_profiles.pdf')

    plt.close()


    internal_step = 0
    ### keep looping over until we reach the next time iteration
    while model_time < nd(t_retro[retro_step]*u.megayear):
        
        temperature_for_D = Tt_path_interp(dim(model_time, u.megayear).m) # T_retro[retro_step]
        pressure_for_D =  Pt_path_interp(dim(model_time, u.megayear).m) # P_retro[retro_step]
        ### determine the diffusion matrix from current temp
        D_term, D_max = update_kappa( temperature_for_D, pressure_for_D)
    
        dt = dt_fac * (mesh.get_min_radius()**2 / D_max)
        
        print(f'\n\ninternal step = {internal_step}, model time = {dim(model_time, u.megayear)}, temperature = {Tt_path_interp(dim(model_time, u.megayear).m)}\n\n')
        ### if model time + dt is less than the next time we want, just use dt
        if model_time+dt < nd(t_retro[retro_step]*u.megayear):
            dt = dt
        else:
            ### If it exceeds the time we're solving over we determine dt to the difference 
            dt = (nd(t_retro[retro_step]*u.megayear) - model_time)

        with mesh.access(Ca, Fe, Mg, Mn):
            Ca.data[:,0] = 1 - Fe.data[:,0] - Mg.data[:,0] - Mn.data[:,0] 

        
        flux_term_sym = D_term[:3,:3] * grad_matrix
        flux_terms = flux_term_sym.subs( {nabla_Fe:gradient_calc(Fe.sym[0]), nabla_Mg:gradient_calc(Mg.sym[0]), nabla_Mn:gradient_calc(Mn.sym[0]) } )
    
        Fe_flux_term = flux_terms[0]
        Mg_flux_term = flux_terms[1]
        Mn_flux_term = flux_terms[2]

        
        
        ### solve the equation
        solve_diff_eq(diff_Fe, Fe, Fe_star, Fe_flux_term, Fe_flux_star, dt)
        solve_diff_eq(diff_Mg, Mg, Mg_star, Mg_flux_term, Mg_flux_star, dt)
        solve_diff_eq(diff_Mn, Mn, Mn_star, Mn_flux_term, Mn_flux_star, dt)

        ### save history terms
        Fe_flux_star = Fe_flux_term.copy()
        Mg_flux_star = Mg_flux_term.copy()
        Mn_flux_star = Mn_flux_term.copy()


        update_history_terms()


        model_time += dt
        internal_step += 1


    ### move up a step after reaching the time where the next shell grows
    step += 1
    retro_step += 1

# %%
mesh.petsc_save_checkpoint(step, meshVars=[Fe, Mg, Mn, Ca], outputPath=outputDir)

# %%
Fe_profile_1 = uw.function.evaluate(Fe.sym[0], profile_coords)
Mg_profile_1 = uw.function.evaluate(Mg.sym[0], profile_coords)
Mn_profile_1 = uw.function.evaluate(Mn.sym[0], profile_coords)
Ca_profile_1 = uw.function.evaluate(Ca.sym[0], profile_coords)

plt.plot(profile_coords[:,0]*Rr.max(), Fe_profile_1, c='blue')
plt.plot(profile_coords[:,0]*Rr.max(), Mg_profile_1, c='red')
plt.plot(profile_coords[:,0]*Rr.max(), Mn_profile_1, c='green')
plt.plot(profile_coords[:,0]*Rr.max(), Ca_profile_1, c='orange')

plt.plot(Rr, Fer, label='Fe', ls=':', c='blue')
plt.plot(Rr, Mgr, label='Mg', ls=':', c='red')
plt.plot(Rr, Mnr, label='Mn', ls=':', c='green')
plt.plot(Rr, Car, label='Ca', ls=':', c='orange')

plt.title(f'time = {round(dim(model_time, u.megayear).m, 2)} Myr')

plt.xlabel('r [$\mu m$]')
plt.ylabel('F')


plt.grid()
plt.legend()

plt.savefig(f'{outputDir}step.{step}.diffusion_profiles.pdf')

plt.close()

# %%
CZGM_EM_data_final = pd.read_csv('./CZGM_data/combined_C_final_cart.csv', header=None)


# %%
CZGM_EM_data_final

# %%
# Fe_profile_Final = uw.function.evaluate(Fe.sym[0], profile_coords)
# Mg_profile_Final = uw.function.evaluate(Mg.sym[0], profile_coords)
# Mn_profile_Final = uw.function.evaluate(Mn.sym[0], profile_coords)
# Ca_profile_Final = uw.function.evaluate(Ca.sym[0], profile_coords)


# plt.plot(profile_coords[:,0]*Rr.max(), Fe_profile_EndGrowth, c='blue', lw=0.5)
# plt.plot(profile_coords[:,0]*Rr.max(), Mg_profile_EndGrowth, c='red', lw=0.5)
# plt.plot(profile_coords[:,0]*Rr.max(), Mn_profile_EndGrowth, c='green', lw=0.5)
# plt.plot(profile_coords[:,0]*Rr.max(), Ca_profile_EndGrowth, c='orange', lw=0.5)


plt.plot(profile_coords[:,0]*Rr.max(), Fe_profile_1, c='red', label='Fe')
plt.plot(profile_coords[:,0]*Rr.max(), Mg_profile_1, c='green', label='Mg')
plt.plot(profile_coords[:,0]*Rr.max(), Mn_profile_1, c='blue', label='Mn')
plt.plot(profile_coords[:,0]*Rr.max(), Ca_profile_1, c='gold', label='Ca')


plt.plot(Rr, Fer, ls=':', c='red')
plt.plot(Rr, Mgr, ls=':', c='green')
plt.plot(Rr, Mnr, ls=':', c='blue')
plt.plot(Rr, Car, ls=':', c='gold')

plt.scatter(CZGM_EM_data_final.iloc[:,0], CZGM_EM_data_final.iloc[:,1], c='red', marker='x')
plt.scatter(CZGM_EM_data_final.iloc[:,0], CZGM_EM_data_final.iloc[:,2], c='green', marker='x')
plt.scatter(CZGM_EM_data_final.iloc[:,0], CZGM_EM_data_final.iloc[:,3], c='blue', marker='x')
plt.scatter(CZGM_EM_data_final.iloc[:,0], CZGM_EM_data_final.iloc[:,4], c='gold', marker='x')
 

plt.title(f'time = {round(dim(model_time, u.megayear).m, 2)} Myr')


plt.grid()
plt.legend()

plt.savefig(f'{outputDir}final_TS_CZGM_comp_annulus_cyldricalCoords.pdf')


plt.show()
# %%

# %%

# %%
