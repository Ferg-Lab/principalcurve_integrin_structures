"""
Generated by CHARMM-GUI (http://www.charmm-gui.org)

omm_readinputs.py

This module is for reading inputs in OpenMM.

Correspondance: jul316@lehigh.edu or wonpil@lehigh.edu
Last update: March 29, 2017
"""

from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *
import time

class _OpenMMReadInputs():

    def __init__(self):
        self.gen_vel          = 'no'                                      # Generate initial velocities
        self.gen_temp         = 310.0                                     # Temperature for generating initial velocities (K)
        self.gen_seed         = None                                      # Seed for generating initial velocities
        self.nstep            = 0                                         # Number of steps to run
        self.dt               = 0.002                                     # Time-step (ps)

        self.nstout           = 100                                       # Writing output frequency (steps)
        self.nstdcd           = 0                                         # Wrtiing coordinates trajectory frequency (steps)

        self.coulomb          = PME                                       # Electrostatic cut-off method
        self.ewald_Tol        = 0.0005                                    # Ewald error tolerance
        self.vdw              = 'Force-switch'                            # vdW cut-off method
        self.r_on             = 1.0                                       # Switch-on distance (nm)
        self.r_off            = 1.2                                       # Switch-off distance (nm)

        self.temp             = 310.0                                     # Temperature (K)
        self.fric_coeff       = 1                                         # Friction coefficient for Langevin dynamics

        self.pcouple          = 'no'                                      # Turn on/off pressure coupling
        self.p_ref            = 1.0                                       # Pressure (Pref or Pxx, Pyy, Pzz; bar)
        self.p_type           = 'membrane'                                # MonteCarloBarotat type
        self.p_scale          = True, True, True                          # For MonteCarloAnisotropicBarostat
        self.p_XYMode         = MonteCarloMembraneBarostat.XYIsotropic    # For MonteCarloMembraneBarostat
        self.p_ZMode          = MonteCarloMembraneBarostat.ZFree          # For MonteCarloMembraneBarostat
        self.p_tens           = 0.0                                       # Sulface tension for MonteCarloMembraneBarostat (dyne/cm)
        self.p_freq           = 15                                        # Pressure coupling frequency (steps)

        self.cons             = HBonds                                    # Constraints method

        self.rest             = 'no'                                      # Turn on/off restraints

    def read(self, inputFile):
        for line in open(inputFile, 'r'):
            if line.find('#') >= 0: line = line.split('#')[0]
            line = line.strip()
            if len(line) > 0:
                segments = line.split('=')
                input_param = segments[0].upper().strip()
                try:    input_value = segments[1].strip()
                except: input_value = None
                if input_value:
                    if input_param == 'GEN_VEL':
                        if input_value.upper() == 'YES':                self.gen_vel          = 'yes'
                        if input_value.upper() == 'NO':                 self.gen_vel          = 'no'
                    if input_param == 'GEN_TEMP':                       self.gen_temp         = float(input_value)
                    if input_param == 'GEN_SEED':                       self.gen_seed         = int(input_value)
                    if input_param == 'NSTEP':                          self.nstep            = int(input_value)
                    if input_param == 'DT':                             self.dt               = float(input_value)
                    if input_param == 'NSTOUT':                         self.nstout           = int(input_value)
                    if input_param == 'NSTDCD':                         self.nstdcd           = int(input_value)
                    if input_param == 'COULOMB':
                        if input_value.upper() == 'NOCUTOFF':           self.coulomb          = NoCutoff
                        if input_value.upper() == 'CUTOFFNONPERIODIC':  self.coulomb          = CutoffNonPeriodic
                        if input_value.upper() == 'CUTOFFPERIODIC':     self.coulomb          = CutoffPeriodic
                        if input_value.upper() == 'EWALD':              self.coulomb          = Ewald
                        if input_value.upper() == 'PME':                self.coulomb          = PME
                    if input_param == 'EWALD_TOL':                      self.ewald_Tol        = float(input_value)
                    if input_param == 'VDW':
                        if input_value.upper() == 'NOCUTOFF':           self.vdw              = 'NoCutoff'
                        if input_value.upper() == 'CUTOFFPERIODIC':     self.vdw              = 'CutoffPeriodic'
                        if input_value.upper() == 'FORCE-SWITCH':       self.vdw              = 'Force-switch'
                        if input_value.upper() == 'SWITCH':             self.vdw              = 'Switch'
                        if input_value.upper() == 'LJPME':              self.vdw              = 'LJPME'
                    if input_param == 'R_ON':                           self.r_on             = float(input_value)
                    if input_param == 'R_OFF':                          self.r_off            = float(input_value)
                    if input_param == 'TEMP':                           self.temp             = float(input_value)
                    if input_param == 'FRIC_COEFF':                     self.fric_coeff       = float(input_value)
                    if input_param == 'PCOUPLE':
                        if input_value.upper() == 'YES':                self.pcouple          = 'yes'
                        if input_value.upper() == 'NO':                 self.pcouple          = 'no'
                    if input_param == 'P_REF':
                        if input_value.find(',') < 0:
                            self.p_ref = float(input_value)
                        else:
                            Pxx = float(input_value.split(',')[0])
                            Pyy = float(input_value.split(',')[1])
                            Pzz = float(input_value.split(',')[2])
                            self.p_ref = Pxx, Pyy, Pzz
                    if input_param == 'P_TYPE':
                        if input_value.upper() == 'ISOTROPIC':          self.p_type           = 'isotropic'
                        if input_value.upper() == 'MEMBRANE':           self.p_type           = 'membrane'
                        if input_value.upper() == 'ANISOTROPIC':        self.p_type           = 'anisotropic'
                    if input_param == 'P_SCALE':
                        scaleX = True
                        scaleY = True
                        scaleZ = True
                        if input_value.upper().find('X') < 0: scaleX = False
                        if input_value.upper().find('Y') < 0: scaleY = False
                        if input_value.upper().find('Z') < 0: scaleZ = False
                        self.p_scale = scaleX, scaleY, scaleZ
                    if input_param == 'P_XYMODE':
                        if input_value.upper() == 'XYISOTROPIC':        self.p_XYMode         = MonteCarloMembraneBarostat.XYIsotropic
                        if input_value.upper() == 'XYANISOTROPIC':      self.p_XYMode         = MonteCarloMembraneBarostat.XYAnisotropic
                    if input_param == 'P_ZMODE':
                        if input_value.upper() == 'ZFREE':              self.p_ZMode          = MonteCarloMembraneBarostat.ZFree
                        if input_value.upper() == 'ZFIXED':             self.p_ZMode          = MonteCarloMembraneBarostat.ZFixed
                        if input_value.upper() == 'CONSTANTVOLUME':     self.p_ZMode          = MonteCarloMembraneBarostat.ConstantVolume
                    if input_param == 'P_TENS':                         self.p_tens           = float(input_value)
                    if input_param == 'P_FREQ':                         self.p_freq           = int(input_value)
                    if input_param == 'CONS':
                        if input_value.upper() == 'NONE':               self.cons             = None
                        if input_value.upper() == 'HBONDS':             self.cons             = HBonds
                        if input_value.upper() == 'ALLBONDS':           self.cons             = AllBonds
                        if input_value.upper() == 'HANGLES':            self.cons             = HAngles

        return self

def read_inputs(inputFile):
    return _OpenMMReadInputs().read(inputFile)


def read_params(filename):
    parFiles = ()
    for line in open(filename, 'r'):
        if '!' in line: line = line.split('!')[0]
        parfile = line.strip()
        if len(parfile) != 0: parFiles += ( parfile, )

    params = CharmmParameterSet( *parFiles )
    return params

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

