from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from simtk import openmm
from sys import stdout
import sys
#sys.path.append('seekr2_openmm_plugin/')
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



print("platform failures")
print(Platform.getPluginLoadFailures())
print("\n")

# HELPER FUNCTIONS
def simple_distance(point1, point2):#, box_size):
    # Calculate the distance vector between the two points
    d = point2 - point1

    # Apply periodic boundary conditions
    #d = np.where(np.abs(d) > 0.5 * box_size, d - np.sign(d) * box_size, d)

    # Calculate the distance
    distance = np.linalg.norm(d)

    return distance

def get_l_alpha(total_length, alpha, N):
    
    return total_length * ((alpha)/N)

def get_cap_l_alpha(newimages, alpha):
    
    N = alpha+1
    total_length = 0
    for i in range(0,N):
        if i!=0:
         total_length += simple_distance(newimages[i],newimages[i-1])
    
    return total_length

def reparameterize(newimages, alpha):

    N = len(newimages)-1
    total_length = get_cap_l_alpha(newimages, N)
        
    l_alpha = get_l_alpha(total_length, alpha, N)
    
    # find a_alpha based on the inequality
    a_alpha = alpha # set default
    for i in range(1,N):
        capL_alpha_minus_1 = get_cap_l_alpha(newimages, i-1)
        capL_alpha = get_cap_l_alpha(newimages, i)
        
        if (capL_alpha_minus_1 < l_alpha) and (l_alpha <= capL_alpha):
            a_alpha = i
            break
    
    #print(f'a_alpha:{a_alpha}, alpha:{alpha}')
    
    capL_alpha_minus_1 = 0
    for i in range(a_alpha):
        if i !=0:
            capL_alpha_minus_1 += simple_distance(newimages[i],newimages[i-1])
        
    if alpha == 0 or alpha == len(newimages)-1:
        #print(f'alpha: {alpha}')
        reparameterized_image = newimages[alpha]
    else:
        diff = newimages[a_alpha] - newimages[a_alpha-1]
        #print(f'diff: {diff}')
        diff_distance = simple_distance(newimages[a_alpha], newimages[a_alpha-1])
        reparameterized_image = newimages[a_alpha-1] + (l_alpha - capL_alpha_minus_1)*((diff)/(diff_distance))

    
    return reparameterized_image

def find_closest_index(tuple_data, target_point):
    closest_index = None
    closest_distance = float('inf')
    
    for i, element in enumerate(tuple_data):
        if i == 0: # SKIP FIRST PDB FRAME
            continue
        distance = math.dist(element, target_point)
        if distance < closest_distance:
            closest_distance = distance
            closest_index = i
    
    return closest_index


# GLOBAL SETTINGS
STARTITERATION=55 #146 #124 #8 #40 #30 #21 #20 #10 #2
NITERATIONS=60 #151 #10 #50 #40 #30 #20 #10
NIMAGES=19
simlocation="/project2/andrewferguson/sivadasetty/doe/analysis-integrin/string_mechanisms/deadbolt/string_parallel_100k_full_lipid_case/final_string_4/"
#simlocation="/home/sivadasetty/scratch-midway2/doe/string_parallel_100k_full_lipid_case/final_string/"

ag1_indices = 'index 14621 14756 14821 14918 15001 15087 15173 15213 15239 15273 15288 15312 15327 15342'
ag2_indices = 'index 25941 26024 26101 26204 26290 26357 26429 26502 26570 26608 26655 26713 26737 26762 26796 26829 26864 26895'
ag3_indices = 'index 57 149 253 379 465 550 621 686 761 865 928 1019 1135 1224 1381 1513 1645 1789 1896 2021 2139 2241 2365 2519 2657 2739 2812 2907 2984 3053 3116 3218 3325 3425 3512 3596 3711 3859 3958 4044 4145 4248 4338 4414 4481 4542 4621 4706 4804 4881 4929 5029 5141 5231 5361 5540 5742 5871 5983 6119 6230 6328 6428 6527 6600'
ag4_indices = 'index 24438 24573 24684 24764 24817 24873 24924 24985 25075 25140 25216 25279 25334 25416 25475 25515 25545 25601 25649 25698 25753 25823 25888'
deltat=0.1 # String method update timestep
kappa=0.1  # String method reparameterization smoothening parameter

# THIS IS FINAL STRING FROM THE PREVIOUS STRING CALCULATION
previous_image_centers = np.array([[1.92740418,  6.32755634],
                                   [1.95147825,  7.08935996],
                                   [2.00508391,  7.84965951],
                                   [2.14498217,  8.59887943],
                                   [2.2894458 ,  9.34724976],
                                   [2.45884303, 10.09037394],
                                   [2.73380458, 10.80116016],
                                   [3.03560236, 11.5010476 ],
                                   [3.4204724 , 12.15888834],
                                   [3.87085137, 12.77377672],
                                   [4.35625171, 13.36140748],
                                   [4.88199026, 13.91190964],
                                   [5.57340259, 14.23237167],
                                   [6.29456104, 14.4789135 ],
                                   [7.03169938, 14.67269036],
                                   [7.77419929, 14.844799  ],
                                   [8.52200106, 14.99215441],
                                   [9.27493833, 15.11053705],
                                   [10.03056293,15.21034195]])

print('Checking distances between final string images ... \n')
for i in range(np.array(previous_image_centers).shape[0]):
    for k in range(i+1, np.array(previous_image_centers).shape[0]):
        if k-i == 1:
            print(i, k, simple_distance(previous_image_centers[i], previous_image_centers[k]))

print('iter 0')
print('previous_image_centers')
print(previous_image_centers)
print('\n')

# FIRST COMPUTE THE CURRENT ITERATION BASED PREVIOUS CENTERS OR STRING CENTERS
# NOTE: IN THESE FINAL STRING CALCULATIONS: ITER 1 IS ACTUALLY ITER 0. WE STARTED ITER 1 WITH FINAL STRING FROM LAST STRING.
# SO EVERYTING IS OFFSET BY +1. 
# ALSO THIS PART OF THE STRING CALCULATIONS IS ONLY NEEDED FOR RESTARTING. IF STRINGS ARE NOT SAVED.
# FURTHER; BELOW IT SHOULD BE STARTITERATION RATHER THAN STARTITERATION+1 BECAUSE THE STRINGS ARE AGAIN CALCULATED
# IN THE NEXT LOOP FROM THE FINAL STRING ESTIMATED HERE.
for j in range(1+1,STARTITERATION): #+1):

    simlocation="/project2/andrewferguson/sivadasetty/doe/analysis-integrin/string_mechanisms/deadbolt/string_parallel_100k_full_lipid_case/final_string_4/"
    #simlocation="/home/sivadasetty/scratch-midway2/doe/string_parallel_100k_full_lipid_case/final_string/"

    # Compute the image centers using previous iteration
    path_to_file = simlocation + 'iter' + str(j-1) + '/'

    # initial computation
    print('initial computation')
    print(path_to_file)
    print('\n')

    runningaverages = []
    for i in range(NIMAGES):
        #if j-1 == 0:
        #    temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize.pdb', path_to_file+ 'target_md_' + str(i) + '/output.xtc')
        #else:
            #temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize.pdb', path_to_file+ 'target_md_' + str(i) + '/output.dcd')
        temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize-protein.pdb', path_to_file+ 'target_md_' + str(i) + '/output-protein.dcd')

        ag1 = temp_universe.atoms.select_atoms(ag1_indices)
        ag2 = temp_universe.atoms.select_atoms(ag2_indices)
        ag3 = temp_universe.atoms.select_atoms(ag3_indices)
        ag4 = temp_universe.atoms.select_atoms(ag4_indices)

        avg_x = []
        avg_y = []
        for ts in temp_universe.trajectory[::1]:

            #xdist = np.linalg.norm(ag1.center_of_mass()-ag2.center_of_mass())
            #ydist = np.linalg.norm(ag3.center_of_mass()-ag4.center_of_mass())
            xdist = distances.distance_array(ag1.center_of_geometry(), 
                                             ag2.center_of_geometry(), box = temp_universe.dimensions)
            ydist = distances.distance_array(ag3.center_of_geometry(), 
                                             ag4.center_of_geometry(), box = temp_universe.dimensions)

            avg_x.append(xdist)
            avg_y.append(ydist)

        runningaverages.append([np.mean(avg_x)/10, np.mean(avg_y)/10])

    # update the string using finite temperature string method algorithm
    N=NIMAGES-1 # 0 to N notation; so N in paper is not number of images but - 1 of it.
    kappa_n=kappa*(N+1)*deltat

    new_images_vector_B_2D = []
    for k in range(2):
        new_images_vector_B = np.zeros(N+1)
        for i in range(N+1):

            new_images_vector_B[i] = previous_image_centers[i][k] - deltat*(previous_image_centers[i][k] - runningaverages[i][k])

        new_images_vector_B_2D.append(new_images_vector_B)

    newimages_matrix_A_2D = []
    newimages_matrix_A = np.eye(N+1)
    for i in range(N+1):
    
        if i == 0 or i == N:
            continue
        else:
            newimages_matrix_A[i,i-1]=-kappa_n
            newimages_matrix_A[i,i]=(1+2*kappa_n)
            newimages_matrix_A[i,i+1]=-kappa_n

        newimages_matrix_A_2D.append(newimages_matrix_A)

    #### SOLVE LINEAR EQUATIONS FOR LOCATION OF NEW IMAGES  
    newimages = []
    for k in range(2):
        newimages.append(np.linalg.solve(newimages_matrix_A_2D[k], new_images_vector_B_2D[k]))

    # Add the first and last points with boundaries
    newimages = np.array(list(zip(newimages[0], newimages[1])))

    #### REPARAMTERIZE FOR KEEPING THE IMAGES EQUIDISTANT
    reparameterized_images = []
    for i in range(len(newimages)):
        reparameterized_images.append(reparameterize(newimages, i))

    previous_image_centers = np.array(reparameterized_images)

    print('\n')
    print(f'iter: {j}')
    print('New previous image centers')
    print(previous_image_centers)
    print('\n')


# START THE NEW ITERATION BY FIRST GENERATING THE INITIAL STRUCTURES
for j in range(STARTITERATION,NITERATIONS):

    simlocation="/project2/andrewferguson/sivadasetty/doe/analysis-integrin/string_mechanisms/deadbolt/string_parallel_100k_full_lipid_case/final_string_4/"
    #simlocation="/home/sivadasetty/scratch-midway2/doe/string_parallel_100k_full_lipid_case/final_string/"

    print(STARTITERATION, j)
    print(simlocation)

    # INITIAL STRUCTURE CALCULATIONS
    # Compute the image centers using previous iteration
    path_to_file = simlocation + 'iter' + str(j-1) + '/'

    print(path_to_file)

    runningaverages = []
    for i in range(NIMAGES):

        #if j-1 == 0:
        #    temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize.pdb', path_to_file+ 'target_md_' + str(i) + '/output.xtc')
        #else:
        temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize.pdb', path_to_file+ 'target_md_' + str(i) + '/output.dcd')
        #temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize-protein.pdb', path_to_file+ 'target_md_' + str(i) + '/output-protein.dcd')

    
        ag1 = temp_universe.atoms.select_atoms(ag1_indices)
        ag2 = temp_universe.atoms.select_atoms(ag2_indices)
        ag3 = temp_universe.atoms.select_atoms(ag3_indices)
        ag4 = temp_universe.atoms.select_atoms(ag4_indices)
    
        avg_x = []
        avg_y = []
        for ts in temp_universe.trajectory[::1]: #[::10]
        
            #xdist = np.linalg.norm(ag1.center_of_mass()-ag2.center_of_mass())    
            #ydist = np.linalg.norm(ag3.center_of_mass()-ag4.center_of_mass())
            xdist = distances.distance_array(ag1.center_of_geometry(),
                                             ag2.center_of_geometry(), box = temp_universe.dimensions)
            ydist = distances.distance_array(ag3.center_of_geometry(),
                                             ag4.center_of_geometry(), box = temp_universe.dimensions)
        
            avg_x.append(xdist)
            avg_y.append(ydist)
        
        runningaverages.append([np.mean(avg_x)/10, np.mean(avg_y)/10])

    # RUNNING AVERAGES
    print(f'iter:{j} \t RUNNING AVERAGES \n')
    print(runningaverages)

    # update the string using finite temperature string method algorithm
    N=NIMAGES-1 # 0 to N notation; so N in paper is not number of images but - 1 of it.
    kappa_n=kappa*(N+1)*deltat

    new_images_vector_B_2D = []
    for k in range(2):
        new_images_vector_B = np.zeros(N+1)
        for i in range(N+1):

            new_images_vector_B[i] = previous_image_centers[i][k] - deltat*(previous_image_centers[i][k] - runningaverages[i][k])
        
        new_images_vector_B_2D.append(new_images_vector_B)

    newimages_matrix_A_2D = []
    newimages_matrix_A = np.eye(N+1)
    for i in range(N+1):
    
        if i == 0 or i == N:
            continue
        else:
            newimages_matrix_A[i,i-1]=-kappa_n
            newimages_matrix_A[i,i]=(1+2*kappa_n)
            newimages_matrix_A[i,i+1]=-kappa_n
            
        newimages_matrix_A_2D.append(newimages_matrix_A)

    #### SOLVE LINEAR EQUATIONS FOR LOCATION OF NEW IMAGES  
    newimages = []   
    for k in range(2):
        newimages.append(np.linalg.solve(newimages_matrix_A_2D[k], new_images_vector_B_2D[k]))   

    # Add the first and last points with boundaries
    newimages = np.array(list(zip(newimages[0], newimages[1])))
    
    #### REPARAMTERIZE FOR KEEPING THE IMAGES EQUIDISTANT
    reparameterized_images = []
    for i in range(len(newimages)):
        reparameterized_images.append(reparameterize(newimages, i))

    print(f'iter:{j}\t REPARAMETERIZED IMAGES \n')
    print(reparameterized_images)
    
    print('\nChecking distances between reparameterized images ... \n')
    for i in range(np.array(reparameterized_images).shape[0]):
        for k in range(i+1, np.array(reparameterized_images).shape[0]):
            if k-i == 1:
                print(i, k, simple_distance(reparameterized_images[i], reparameterized_images[k]))


    # EXTRACT STRUCTURES BASED ON CLOSEST AVAILABLE SNAPSHOT TO THE CALCULATED reparameterized_images;
    reparameterized_frame_indices = {}
    for i in range(NIMAGES):
        # READ WHOLE STRUCTURES. [FUTURE: SAVE DISTANCES AND FRAMES FROM PREVIOUS CALCULATION AS DICTIONARY]

        #if j-1 == 0:
        #    temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize.pdb', path_to_file+ 'target_md_' + str(i) + '/output.xtc')
        #else:
            #temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize.pdb', path_to_file+ 'target_md_' + str(i) + '/output.dcd')
        #temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize-protein.pdb', path_to_file+ 'target_md_' + str(i) + '/output-protein.dcd')

        temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize.pdb', path_to_file+ 'target_md_' + str(i) + '/output.dcd')


        ag1 = temp_universe.atoms.select_atoms(ag1_indices)
        ag2 = temp_universe.atoms.select_atoms(ag2_indices)
        ag3 = temp_universe.atoms.select_atoms(ag3_indices)
        ag4 = temp_universe.atoms.select_atoms(ag4_indices)
    
        avg_x = []
        avg_y = []
        for ts in temp_universe.trajectory[::1]: #[::10]
            
            #xdist = np.linalg.norm(ag1.center_of_mass()-ag2.center_of_mass())/10
            #ydist = np.linalg.norm(ag3.center_of_mass()-ag4.center_of_mass())/10
            
            xdist = distances.distance_array(ag1.center_of_geometry(),
                                             ag2.center_of_geometry(), box = temp_universe.dimensions)/10
            ydist = distances.distance_array(ag3.center_of_geometry(),
                                             ag4.center_of_geometry(), box = temp_universe.dimensions)/10
            
            avg_x.append(xdist)
            avg_y.append(ydist)
        
    
        tuple_data = tuple(zip(avg_x, avg_y))
        fr_index = find_closest_index(tuple_data, reparameterized_images[i])
        print(f'\n fr_index: {fr_index} \n')
        reparameterized_frame_indices[i] = fr_index


    # NEW IMAGE SNAPSHOTS
    outputpath = simlocation + 'iter' + str(j) + '/' + 'initial_frames'
    if not os.path.exists(simlocation + 'iter' + str(j)):
        os.makedirs(simlocation + 'iter' + str(j))
    
    if not os.path.exists(simlocation + 'iter' + str(j) + '/' + 'initial_frames'):
        os.makedirs(simlocation + 'iter' + str(j) + '/' + 'initial_frames')

    for i in range(NIMAGES):
        # READ WHOLE SYSTEM; INCLUDING BILAYER
        #temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize_all.pdb', path_to_file+ 'target_md_' + str(i) + '/output_all.xtc')
        #temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize.pdb', path_to_file+ 'target_md_' + str(i) + '/output.dcd')

        #if j-1 == 0:
        #    temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize_all.pdb', path_to_file+ 'target_md_' + str(i) + '/output_all.xtc')
        #else:
            #temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize.pdb', path_to_file+ 'target_md_' + str(i) + '/output.dcd')
        temp_universe = mda.Universe(path_to_file+ 'target_md_' + str(i) + '/minimize.pdb', path_to_file+ 'target_md_' + str(i) + '/output.dcd')



        temp_universe.trajectory[reparameterized_frame_indices[i]]
        temp_universe.select_atoms("all").write(outputpath+"/reparameterized_image_"+str(i)+".pdb")
        temp_universe.select_atoms("all").write(outputpath+"/reparameterized_image_"+str(i)+".gro")
    
    print('reparameterized_images')
    print(reparameterized_images)
    print('\n')


    #exit()


    # SANITY CHECK: DISTANCE BETWEEN FULL STRUCTURE SNAPSHOTS CORRESPONDING TO REPARAMETERIZED IMAGES
    
    extracted_images_distances = []
    for i in range(NIMAGES):
        temp_universe = mda.Universe(outputpath+"/reparameterized_image_"+str(i)+".pdb")

        ag1 = temp_universe.atoms.select_atoms(ag1_indices)
        ag2 = temp_universe.atoms.select_atoms(ag2_indices)
        ag3 = temp_universe.atoms.select_atoms(ag3_indices)
        ag4 = temp_universe.atoms.select_atoms(ag4_indices)

        xdist = distances.distance_array(ag1.center_of_geometry(),
                                         ag2.center_of_geometry(), box = temp_universe.dimensions)
        ydist = distances.distance_array(ag3.center_of_geometry(),
                                         ag4.center_of_geometry(), box = temp_universe.dimensions)

        extracted_images_distances.append(np.array([xdist/10, ydist/10]))

    print('Distances between extracted full structure snapshots closest to reparameterized images ... \n')
    for i in range(np.array(extracted_images_distances).shape[0]):
        for k in range(i+1, np.array(extracted_images_distances).shape[0]):
            if k-i == 1:
                print(i, k, simple_distance(extracted_images_distances[i], extracted_images_distances[k]))



    # RUN EACH ITERATION OF STRING METHOD -- STARTING AFTER THE INITIAL GAN GENERATED CALCULATIONS
    simlocation+="iter"+str(j)+"/"

    # RUN EACH IMAGE
    batch_submission_file = "array"
    subprocess.call(f"sbatch -J job_{j} {batch_submission_file}.sbatch {simlocation} '{reparameterized_images}' {j}", shell=True)

    batch_submission_file = "array2"
    subprocess.call(f"sbatch -J job_{j} {batch_submission_file}.sbatch {simlocation} '{reparameterized_images}' {j}", shell=True)

    batch_submission_file = "array3"
    subprocess.call(f"sbatch -J job_{j} --wait {batch_submission_file}.sbatch {simlocation} '{reparameterized_images}' {j}", shell=True)

    #batch_submission_file = "array4"
    #subprocess.call(f"sbatch -J job_{j} --wait {batch_submission_file}.sbatch {simlocation} '{reparameterized_images}' {j}", shell=True)



    # update previous image centers
    previous_image_centers = np.array(reparameterized_images) 

    # report
    print(f'updated previous image centers used for starting cycle: {j}...')
    print(previous_image_centers)
    print('\n')



