import sys
import os

from omm_readinputs import *
from omm_barostat import *

from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from sys import stdout
from seekr2plugin import MmvtLangevinIntegrator, MmvtLangevinMiddleIntegrator
import seekr2plugin
from time import time
import numpy as np
import os
import mdtraj
import MDAnalysis as mda
from mdareporter import MDAReporter
from MDAnalysis.analysis import distances
import parmed as pmd
import time
import psutil
import pickle
import subprocess
import multiprocessing
from multiprocessing import Pool
import sys
import json
import ast


# HELPER FUNCTIONS
def compute_center_of_geometry(positions, indices):
    cog = np.zeros(3)
    for i in indices:
        cog += np.array(positions[i])
    cog /= len(indices)
    return cog

# GLOBAL SETTINGS
k=1.0*kilojoules_per_mole # Voronoi force
BOUNCEFILE = 'bounce_integrin.txt'
STATEFOLDER = 'bounce_state'
STATEFILE = 'state'
COMLOGFREQUENCY =  100
#NSTEPS = 500000
NSTEPS = 100000
COMLOGFILE = 'cv_cvdistances.pkl'
INPUTPARAMETERS='inputs.inp'

# TEMPORARY SETTINGS FOR DEBUGGING
#simlocation="/project2/andrewferguson/sivadasetty/doe/analysis-integrin/string_mechanisms/deadbolt/string_parallel_100k_full_lipid_case/charmmgui_structures_full/"
#DDFF="charmm-gui-0-0964384658"


def main():

    # Check if the correct number of arguments is provided
    print(len(sys.argv), sys.argv)
    if len(sys.argv) != 62:
        print(f"Usage: {len(sys.argv)} python run_each_image.py <number1>")
        #return


    # Set image value
    i = int(sys.argv[1])
    print(f'Image: {i}')

    if i == 0:
        DDFF = str(i)+'-1902399944'
    elif i == 1:
        DDFF = str(i)+'-1902403462'
    elif i == 2:
        DDFF = str(i)+'-1902405186'
    elif i == 3:
        DDFF = str(i)+'-1902406669'
    elif i == 4:
        DDFF = str(i)+'-1902417678'
    elif i == 5:
        DDFF = str(i)+'-1902418574'
    elif i == 6:
        DDFF = str(i)+'-1902420884'
    elif i == 7:
        DDFF = str(i)+'-1902424208'
    elif i == 8:
        DDFF = str(i)+'-1901916785'
    elif i == 9:
        DDFF = str(i)+'-1901923546'
    elif i == 10:
        DDFF = str(i)+'-1958724456'
    elif i == 11:
        DDFF = str(i)+'-1906915548'
    elif i == 12:
        DDFF = str(i)+'-1901928409'
    elif i == 13:
        DDFF = str(i)+'-1901929204'
    elif i == 14:
        DDFF = str(i)+'-1901930278'
    elif i == 15:
        DDFF = str(i)+'-1901931084'
    elif i == 16:
        DDFF = str(i)+'-1901931841'
    elif i == 17:
        DDFF = str(i)+'-1901932850'
    elif i == 18:
        DDFF = str(i)+'-1901933772'


    simlocation = sys.argv[2] 
    print(simlocation)

    print(DDFF)

    # Get reparameterized image locations
    reparameterized_images = sys.argv[3:-1] # json.loads(sys.argv[3:-1])
    reparameterized_images = ''.join(reparameterized_images).replace('array','').replace('[','').replace(']','').replace('(','').replace(')','').split(',')
    reparameterized_images = [(float(reparameterized_images[i]), float(reparameterized_images[i+1])) for i in range(0, len(reparameterized_images), 2)]
    print('check read reparameterized_images')
    print(reparameterized_images)
    print(reparameterized_images[0])
    print('end check \n')

    x_image_centers = {}
    y_image_centers = {}
    for k in range(1,20):
        x_image_centers['x'+str(k)+'1'] = reparameterized_images[k-1][0]
        y_image_centers['x'+str(k)+'2'] = reparameterized_images[k-1][1]

    # LOAD SIMULATION PARAMETERS
    print("Loading parameters")
    inputs = read_inputs(simlocation + '/../' + INPUTPARAMETERS)


    for key, value in vars(inputs).items():
        print(key, ":", value)
    
    # Build system
    nboptions = dict(nonbondedMethod=inputs.coulomb,
                     nonbondedCutoff=inputs.r_off*nanometers,
                     switchDistance=inputs.r_on*nanometers,
                     constraints=inputs.cons,
                     ewaldErrorTolerance=inputs.ewald_Tol)

    ##Get PSF
    print('Loading universe ...')
    project2location="/project2/andrewferguson/sivadasetty/doe/analysis-integrin/string_mechanisms/deadbolt/string_parallel_100k_full_lipid_case/"
    #u = mda.Universe(simlocation + '/../../charmmgui_structures_full_final_string/charmm-gui-' + DDFF + '/gromacs/topol_tip3.top', topology_format='ITP')
    u = mda.Universe(project2location + '/charmmgui_structures_full_final_string/charmm-gui-' + DDFF + '/gromacs/topol_tip3.top', topology_format='ITP')


    print('Loading positions ...')
    #simlocation += '/iter' + str(sys.argv[-1]) + '/'
    #print(simlocation)
    u.load_new(simlocation + "/initial_frames/" + 'reparameterized_image_'+str(i)+'.pdb')

    print('Bonds in universe...')
    print(u.bonds)

    print('Writing PSF ...')
    if not os.path.exists(simlocation + 'target_md_' + str(i)):
        os.makedirs(simlocation + 'target_md_' + str(i))

    if not os.path.exists(simlocation + "target_md_" + str(i) + '/' + "init_" + str(i) + "_aa.psf"):
        u.atoms.convert_to('PARMED').save(simlocation + "target_md_" + str(i) + '/' + "init_" + str(i) + "_aa.psf")

    ## Load PDB and top files
    #print('Loading gro ...')
    gro_universe = mda.Universe(simlocation + "/initial_frames/" + 'reparameterized_image_'+str(i)+'.gro')
    gro = GromacsGroFile(simlocation + "/initial_frames/" + 'reparameterized_image_'+str(i)+'.gro')

    print('Loading CHARMM topology ...')

    top = CharmmPsfFile(simlocation + "target_md_" + str(i) + '/' + "init_" + str(i) + "_aa.psf")
    boxlx=gro.getUnitCellDimensions()[0]
    boxly=gro.getUnitCellDimensions()[1]
    boxlz=gro.getUnitCellDimensions()[2]
    top.setBox(boxlx, boxly, boxlz)


    print('Loaded topology ... ')
    print(top.topology)
    print(top.boxLengths)
    print('\n')


    print('Loading CHARMM parameters ...')
    params = read_params(project2location + '/charmmgui_structures_full_final_string/charmm-gui-' + DDFF + '/openmm/toppar.str')
    #params = read_params(simlocation + '/../../charmmgui_structures_full_final_string/charmm-gui-' + DDFF + '/openmm/toppar.str')


    # Create the OpenMM system
    process = psutil.Process(os.getpid())
    with Timer() as total_time:

        print('Creating OpenMM System ...')
        print('\n')

        system = top.createSystem(params, **nboptions)
        print(system)

        if inputs.pcouple == 'yes':      system = barostat(system, inputs)

        for fi,f in enumerate(system.getForces()):
            print(f.getName(), f.getForceGroup())
            f.setForceGroup(0)

        print('\n')
        print('Creating voronoi groups ...')

        voronoi_expression = "step( k*("
        voronoi_expression += "((distance(g1,g2)-x11)^2 + (distance(g3,g4)-x12)^2) "
        voronoi_expression += "-"
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "min("
        voronoi_expression += "((distance(g1,g2)-x21)^2 + (distance(g3,g4)-x22)^2),"
        voronoi_expression += "((distance(g1,g2)-x31)^2 + (distance(g3,g4)-x32)^2)), "
        voronoi_expression += "((distance(g1,g2)-x41)^2 + (distance(g3,g4)-x42)^2)),"
        voronoi_expression += "((distance(g1,g2)-x51)^2 + (distance(g3,g4)-x52)^2)),"
        voronoi_expression += "((distance(g1,g2)-x61)^2 + (distance(g3,g4)-x62)^2)),"
        voronoi_expression += "((distance(g1,g2)-x71)^2 + (distance(g3,g4)-x72)^2)),"
        voronoi_expression += "((distance(g1,g2)-x81)^2 + (distance(g3,g4)-x82)^2)),"
        voronoi_expression += "((distance(g1,g2)-x91)^2 + (distance(g3,g4)-x92)^2)),"
        voronoi_expression += "((distance(g1,g2)-x101)^2 + (distance(g3,g4)-x102)^2)),"
        voronoi_expression += "((distance(g1,g2)-x111)^2 + (distance(g3,g4)-x112)^2)),"
        voronoi_expression += "((distance(g1,g2)-x121)^2 + (distance(g3,g4)-x122)^2)),"
        voronoi_expression += "((distance(g1,g2)-x131)^2 + (distance(g3,g4)-x132)^2)),"
        voronoi_expression += "((distance(g1,g2)-x141)^2 + (distance(g3,g4)-x142)^2)),"
        voronoi_expression += "((distance(g1,g2)-x151)^2 + (distance(g3,g4)-x152)^2)),"
        voronoi_expression += "((distance(g1,g2)-x161)^2 + (distance(g3,g4)-x162)^2)),"
        voronoi_expression += "((distance(g1,g2)-x171)^2 + (distance(g3,g4)-x172)^2)),"
        voronoi_expression += "((distance(g1,g2)-x181)^2 + (distance(g3,g4)-x182)^2)),"
        voronoi_expression += "((distance(g1,g2)-x191)^2 + (distance(g3,g4)-x192)^2))"
        voronoi_expression += " ))"

        voronoi_force = CustomCentroidBondForce(4,voronoi_expression)


# alpha tail
        mygroup1_indices = [14621,14756,14821,14918,15001,15087,15173,15213,15239,15273,15288,15312,15327,15342]
        mygroup1 = voronoi_force.addGroup(mygroup1_indices,np.ones(len(mygroup1_indices)))

        # beta tail
        mygroup2_indices = [25941,26024,26101,26204,26290,26357,26429,26502,26570,26608,26655,26713,26737,26762,26796,26829,26864,26895]
        mygroup2 = voronoi_force.addGroup(mygroup2_indices,np.ones(len(mygroup2_indices)))

        # beta propeller
        mygroup3_indices = [57,149,253,379,465,550,621,686,761,865,928,1019,1135,1224,1381,1513,1645,1789,1896,2021,2139,2241,2365,2519,2657,2739,2812,2907,2984,3053,3116,3218,3325,3425,3512,3596,3711,3859,3958,4044,4145,4248,4338,4414,4481,4542,4621,4706,4804,4881,4929,5029,5141,5231,5361,5540,5742,5871,5983,6119,6230,6328,6428,6527,6600]
        mygroup3 = voronoi_force.addGroup(mygroup3_indices,np.ones(len(mygroup3_indices)))

        # beta T domain
        mygroup4_indices = [24438,24573,24684,24764,24817,24873,24924,24985,25075,25140,25216,25279,25334,25416,25475,25515,25545,25601,25649,25698,25753,25823,25888]
        mygroup4 = voronoi_force.addGroup(mygroup4_indices,np.ones(len(mygroup4_indices)))


        voronoi_force.setForceGroup(1)
        voronoi_force.addPerBondParameter('k')

        voronoi_force.addPerBondParameter('x11')
        voronoi_force.addPerBondParameter('x12')
        voronoi_force.addPerBondParameter('x21')
        voronoi_force.addPerBondParameter('x22')
        voronoi_force.addPerBondParameter('x31')
        voronoi_force.addPerBondParameter('x32')
        voronoi_force.addPerBondParameter('x41')
        voronoi_force.addPerBondParameter('x42')
        voronoi_force.addPerBondParameter('x51')
        voronoi_force.addPerBondParameter('x52')
        voronoi_force.addPerBondParameter('x61')
        voronoi_force.addPerBondParameter('x62')
        voronoi_force.addPerBondParameter('x71')
        voronoi_force.addPerBondParameter('x72')
        voronoi_force.addPerBondParameter('x81')
        voronoi_force.addPerBondParameter('x82')
        voronoi_force.addPerBondParameter('x91')
        voronoi_force.addPerBondParameter('x92')
        voronoi_force.addPerBondParameter('x101')
        voronoi_force.addPerBondParameter('x102')
        voronoi_force.addPerBondParameter('x111')
        voronoi_force.addPerBondParameter('x112')
        voronoi_force.addPerBondParameter('x121')
        voronoi_force.addPerBondParameter('x122')
        voronoi_force.addPerBondParameter('x131')
        voronoi_force.addPerBondParameter('x132')
        voronoi_force.addPerBondParameter('x141')
        voronoi_force.addPerBondParameter('x142')
        voronoi_force.addPerBondParameter('x151')
        voronoi_force.addPerBondParameter('x152')
        voronoi_force.addPerBondParameter('x161')
        voronoi_force.addPerBondParameter('x162')
        voronoi_force.addPerBondParameter('x171')
        voronoi_force.addPerBondParameter('x172')
        voronoi_force.addPerBondParameter('x181')
        voronoi_force.addPerBondParameter('x182')
        voronoi_force.addPerBondParameter('x191')
        voronoi_force.addPerBondParameter('x192')

        voronoi_force.setUsesPeriodicBoundaryConditions(True) # Explicit #No for Implicit

        cog_group1 = compute_center_of_geometry(gro.getPositions(asNumpy=True).value_in_unit(nanometer), mygroup1_indices)
        cog_group2 = compute_center_of_geometry(gro.getPositions(asNumpy=True).value_in_unit(nanometer), mygroup2_indices)
        cog_group3 = compute_center_of_geometry(gro.getPositions(asNumpy=True).value_in_unit(nanometer), mygroup3_indices)
        cog_group4 = compute_center_of_geometry(gro.getPositions(asNumpy=True).value_in_unit(nanometer), mygroup4_indices)

        # Calculate initial COM distance
        # Apply PBC. NOT DONE.
        #initial_distance1, initial_distance2 = np.linalg.norm(cog_group1 - cog_group2), np.linalg.norm(cog_group3 - cog_group4)


        initial_distance1 = distances.distance_array(cog_group1, cog_group2, box = gro_universe.dimensions)
        initial_distance2 = distances.distance_array(cog_group3, cog_group4, box = gro_universe.dimensions)


        print(f'Initial COM distance: {initial_distance1} {initial_distance2}')
        print('\n')

        print(' Before reference image center in Voronoi force to see if a conformation is close or far from it ')
        print(f"x_image_centers['x11'], y_image_centers['x12'] : {x_image_centers['x11']}, {y_image_centers['x12']}")
        print(f"x_image_centers['x'+str({i}+1)+'1'], y_image_centers['x'+str({i}+1)+'2']: {x_image_centers['x'+str(i+1)+'1']}, {y_image_centers['x'+str(i+1)+'2']}")

        temp_x11, temp_x12 = x_image_centers['x11'], y_image_centers['x12']
        x_image_centers['x11'], y_image_centers['x12'] = x_image_centers['x'+str(i+1)+'1'], y_image_centers['x'+str(i+1)+'2']
        x_image_centers['x'+str(i+1)+'1'], y_image_centers['x'+str(i+1)+'2'] = temp_x11, temp_x12


        print(' Reference image center in Voronoi force to see if a conformation is close or far from it ')
        print(f"x_image_centers['x11'], y_image_centers['x12'] : {x_image_centers['x11']}, {y_image_centers['x12']}")
        print(f"x_image_centers['x'+str({i}+1)+'1'], y_image_centers['x'+str({i}+1)+'2']: {x_image_centers['x'+str(i+1)+'1']}, {y_image_centers['x'+str(i+1)+'2']}")

        print('\n')

        voronoi_force.addBond([mygroup1, mygroup2, mygroup3, mygroup4], [k] + [e for t in zip([x_image_centers['x'+str(k+1)+'1'] for k in range(19)], [y_image_centers['x'+str(k+1)+'2'] for k in range(19)]) for e in t])

        voronoi_force_group = system.addForce(voronoi_force)

        print('\n')
        for fi,f in enumerate(system.getForces()):
            print(f.getName(), f.getForceGroup())

        print('system created.\n')


        # Set platform
        DEFAULT_PLATFORMS = 'CUDA', 'OpenCL', 'CPU'
        enabled_platforms = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
        platform = 'CUDA'
        for platform in DEFAULT_PLATFORMS:
            if platform in enabled_platforms:
                platform = Platform.getPlatformByName(platform)
                break
        if isinstance(platform, str):
            print("Unable to find any OpenMM platform; exiting".format(args.platform[0]), file=sys.stderr)
            sys.exit(1)
        
        print("Using platform:", platform.getName())
        prop = dict(CudaPrecision='single') if platform.getName() == 'CUDA' else dict()
        prop["DeviceIndex"] = "0,1,2,3"


        if not os.path.exists(simlocation + 'target_md_' + str(i) + '/' + 'bounce_state'):
            os.makedirs(simlocation + 'target_md_' + str(i) + '/' + 'bounce_state')

        integrator = MmvtLangevinIntegrator(inputs.temp*kelvin, inputs.fric_coeff/picosecond, inputs.dt*picoseconds, simlocation + 'target_md_' + str(i) + '/' + BOUNCEFILE)
        integrator.addMilestoneGroup(1)
        #integrator.setSaveStateFileName(simlocation + 'target_md_' + str(i) + '/' + STATEFOLDER+ '/' +STATEFILE)


        # Build simulation context
        simulation = Simulation(top.topology, system, integrator, platform, prop)
        simulation.context.setPositions(gro.positions)
        simulation.context.setPeriodicBoxVectors(gro.getPeriodicBoxVectors()[0], gro.getPeriodicBoxVectors()[1], gro.getPeriodicBoxVectors()[2])


        # Calculate initial CV values
        #cvforce = CustomCVForce('r')
        #cvforce.addCollectiveVariable('r', voronoi_force)
        #system.addForce(cvforce)

        
        # Calculate initial system energy
        print("\nInitial system energy")
        print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

        # Generate initial velocities
        if inputs.gen_vel == 'yes':
            print("\nGenerate initial velocities")
            if inputs.gen_seed:
                simulation.context.setVelocitiesToTemperature(inputs.gen_temp, inputs.gen_seed)
            else:
                simulation.context.setVelocitiesToTemperature(inputs.gen_temp)

        state = simulation.context.getState(getPositions=True, getEnergy=True)
        
        with open(simlocation + 'target_md_' + str(i) + '/' + 'before_minimize.pdb', 'w') as output:
            PDBFile.writeFile(simulation.topology, state.getPositions(), output)

        simulation.minimizeEnergy()

        state = simulation.context.getState(getPositions=True, getEnergy=True)
        with open(simlocation + 'target_md_' + str(i) + '/' + 'minimize.pdb', 'w') as output:
            PDBFile.writeFile(simulation.topology, state.getPositions(), output)

        # Output log
        simulation.reporters.append(DCDReporter(simlocation + 'target_md_' + str(i) + '/' + 'output.dcd', 1000))
        simulation.reporters.append(MDAReporter(simlocation + 'target_md_' + str(i) + '/' + 'output.xtc', 1000))

        simulation.reporters.append(StateDataReporter(simlocation + 'target_md_' + str(i) + '/' + 'energy.out', 1000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=NSTEPS, separator='\t'))

        # Check point
        simulation.reporters.append(CheckpointReporter(simlocation + 'target_md_' + str(i) + '/' + 'checkpnt.chk',5000))

        nparticles = system.getNumParticles()


    print(f'{nparticles:12d} atoms | total: {total_time.interval:8.3f} s | CPU mem {process.memory_info().rss/1024/1024:8.3f} MB')

    print('\nRunning simulation ...')
    for fi, f in enumerate(system.getForces()):
        state = simulation.context.getState(getEnergy=True, groups={fi})
        print(f.getName(), state.getPotentialEnergy())
    print('\n')


    for n in range(NSTEPS):

        simulation.step(1)

        #if n % COMLOGFREQUENCY == 0:

        #    cog_group1 = compute_center_of_geometry(simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True).value_in_unit(nanometer), mygroup1_indices)
        #    cog_group2 = compute_center_of_geometry(simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True).value_in_unit(nanometer), mygroup2_indices)
        #    cog_group3 = compute_center_of_geometry(simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True).value_in_unit(nanometer), mygroup3_indices)
        #    cog_group4 = compute_center_of_geometry(simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True).value_in_unit(nanometer), mygroup4_indices)

        #    # Calculate COG distance
        #    id1, id2 = np.linalg.norm(cog_group1 - cog_group2), np.linalg.norm(cog_group3 - cog_group4)
        #    #id1 = distances.distance_array(cog_group1, cog_group2, box = simulation.context.getPeriodicBoxVectors())
        #    #id2 = distances.distance_array(cog_group3, cog_group4, box = simulation.context.getPeriodicBoxVectors())

        #    comdict = {'CV1': id1, 'CV2': id2, 'VDIST0': (id1-x_image_centers['x11'])**2+(id2-y_image_centers['x12'])**2, 'VDIST1': (id1-x_image_centers['x21'])**2+(id2-y_image_centers['x22'])**2, 'VDIST2': (id1-x_image_centers['x31'])**2+(id2-y_image_centers['x32'])**2}
        #    pickle.dump(comdict, open(simlocation + 'target_md_' + str(i) + '/' + COMLOGFILE,'wb'))



if __name__ == "__main__":
    main()


