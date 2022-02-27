import os,yaml
from matplotlib import pyplot as plt

from psana_interface import *

class AttrDict(dict):
    """Class to convert a dictionary to a class.

    Parameters
    ----------
    dict: dictionary

    """

    def __init__(self, *args, **kwargs):
        """Return a class with attributes equal to the input dictionary."""
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class TimetoolInterface:

    def __init__(self, path_config):

        with open(path_config, "r") as config_file:
            self.config = AttrDict(yaml.safe_load(config_file))

        self.init_job()
        self.init_calib_data()

    def init_job(self):
        """
        Initialize job
        Returns
        -------

        """
        if not os.path.exists(self.config.work_dir):
            os.makedirs(self.config.work_dir)

    def init_calib_data(self):
        """
        Initialize calib data

        Returns
        -------

        """
        self.edge_pos = []
        self.amp = []
        self.laser_time = []

    def retrieve_calib_data(self, save=False, overwrite=False):
        """
        Retrieve the pre-calculated edge position and laser timing from data source.

        Returns
        -------

        """
        calib_data_file = f'{self.config.work_dir}calib_data_r{self.config.calib_run}'
        if os.path.isfile(calib_data_file) and not overwrite:
            data = np.load(calib_data_file)
            self.edge_pos = data[0]
            self.amp = data[1]
            self.laser_time = data[2]
            return

        psi = PsanaInterface(exp=self.config.exp,
                             run=self.config.calib_run,
                             parallel=self.config.parallel,
                             small_data=True)

        epics_store = psi.ds.env().epicsStore()

        for idx, evt in enumerate(psi.ds.events()):
            if(idx==10):break
            self.edge_pos = np.append(self.edge_pos,
                                      epics_store.value(self.config.pv_fltpos))
            self.amp = np.append(self.amp,
                                 epics_store.value(self.config.pv_amp))
            self.laser_time = np.append(self.laser_time,
                                        epics_store.value(self.config.pv_lasertime))

        if save:
            np.save(calib_data_file,
                    np.column_stack((self.edge_pos,
                                     self.amp,
                                     self.laser_time)))

    def plot_calib_data(self, png_output=None):
        """
        Plot

        Returns
        -------

        """
        fig = plt.figure(figsize=(4,4), dpi=80)
        plt.plot(self.edge_pos, self.laser_time,
                 'o', color='black',label='edge position')
        plt.xlabel('pixel edge')
        plt.ylabel('laser delay')
        plt.legend()
        if png_output is None:
            plt.show()
        else:
            plt.savefig(png_output)