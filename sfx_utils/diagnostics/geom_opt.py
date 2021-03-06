import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from sfx_utils.diagnostics.run import RunDiagnostics
from sfx_utils.interfaces.psana_interface import assemble_image_stack_batch
from .ag_behenate import *

class GeomOpt:
    
    def __init__(self, exp, run, det_type):
        self.diagnostics = RunDiagnostics(exp=exp, # experiment name, str
                                          run=run, # run number, int
                                          det_type=det_type) # detector name, str
        
    def opt_distance(self, powder, sample='AgBehenate', center=None, plot=False):
        """
        Estimate the sample-detector distance based on the properties of the powder
        diffraction image. Currently only implemented for silver behenate.
        
        Parameters
        ----------
        powder : str or int
            if str, path to the powder diffraction in .npy format
            if int, number of images from which to compute powder 
        sampe : str
            sample type, currently implemented for AgBehenate only
        center : tuple
            detector center (xc,yc) in pixels. if None, assume assembled image center.
        plot : bool
            if True, visualize results of distance estimation

        Returns
        -------
        distance : float
            estimated sample-detector distance in mm
        """
        if type(powder) == str:
            powder_img = np.load(powder)
        
        elif type(powder) == int:
            print("Computing powder from scratch")
            self.diagnostics.compute_run_stats(n_images=powder, powder_only=True)
            powder_img = assemble_image_stack_batch(self.diagnostics.powders['max'], 
                                                    self.diagnostics.pixel_index_map)
        
        else:
            sys.exit("Unrecognized powder type, expected a path or number")
        
        if sample == 'AgBehenate':
            ag_behenate = AgBehenate()
            distance = ag_behenate.opt_distance(powder_img,
                                                self.diagnostics.psi.estimate_distance(),
                                                self.diagnostics.psi.get_pixel_size(), 
                                                self.diagnostics.psi.get_wavelength(),
                                                center=center,
                                                plot=plot)
            return distance

        else:
            print("Sorry, currently only implemented for silver behenate")
            return -1
