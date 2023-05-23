import numpy as np
from scipyu.stats import linregress
import matplotlib.pyplot as plt
from btx.diagnostics.run import RunDiagnostics
from btx.misc.radial import (radial_profile, pix2q)
from typing import Union
from dataclasses import dataclass

@dataclass
class SAXSProfile:
    q_vals: np.ndarray
    intensity: np.ndarray
    globular: bool = True

@dataclass
class GuinierResult:
    fit_result #: LinregressResult
    I0: float
    Rg: float

class SAXSProfiler:
    """! Class to perform radial integration and SAXS analysis."""

    def __init__(self,
                 expmt: str,
                 run: Union[str, int],
                 detector_type: str,
                 rootdir: str,
                 method: str = "btx"):
        """! Integration class initializer.

        Finds the associated files and computes radial profiles.

        @param expmt (str) Experiment name.
        @param run (str | int) Run number.
        @param detector_type (str) Detector type used during the experiment.
        @param rootdir (str) Root directory for btx processing.
        @param method (str) Which radial integration method to use.
        """
        self.diagnostics = RunDiagnostics(exp=expmt,
                                          run=run,
                                          det_type=detector_type)

        if method == 'btx':
            self._saxsprofile = self.integrate1d_btx(expmt, run, rootdir)

    def integrate1d_btx(self,
                        expmt: str,
                        run: Union[str, int],
                        rootdir: str) -> SAXSProfile:
        """! Perform 1D radial integration using the btx implmentation.

        @param expmt (str) Experiment name.
        @param run (str | int) Run number.
        @param rootdir (str) Root directory for btx processing.
        @return SAXSProfile (SAXSProfile) Object containing q's and intensities.
        """
        center = None # self.get_center()
        distance = 0.089 # self.get_distance()

        self.powder = self.find_powder_btx(expmt=expmt, run=run, rootdir=rootdir)
        # Need to update to get center
        iprofile = radial_profile(self.powder)

        qprofile = pix2q(np.arange(iprofile.shape[0]),
                         self.diagnostics.psi.get_wavelength(),
                         distance,
                         self.diagnostics.psi.get_pixel_size())
        return SAXSProfile(qprofile, iprofile)

    def find_powder_btx(self,
                        expmt: str,
                        run: Union[str, int],
                        rootdir: str) -> np.ndarray:
        """! Retrieve the "powder" pattern to be used for radial integration.

        The powder pattern is defined to be a sum over the detector images of
        a run. Assumes a numpy array exists given the directory hierarchy used
        by btx. If no array can be found, it will compute the powder pattern
        using btx's run diagnostics routines.

        @param expmt (str) Experiment name.
        @param run (str | int) Run number.
        @param rootdir (str) Root directory for btx processing.
        @return powder (np.ndarray) 2-D unassembled powder image.
        """
        try:
            path = f'{rootdir}/powder/r{int(run):04}_avg.npy'
            powder = np.load(path)
            return powder
        except FileNotFoundError as e:
            print('Cannot find a powder image, attempting to recompute. '
                  'This my take a while.')

        try:
            self.diagnostics.compute_run_stats(powder_only=True)
            powder = self.diagnostics.powders['sum']
            return powder
        except Exception as e:
            print(f'Unanticipated error: {e}')

    def get_center(self):
        """! Determine the beam center based on the current geometry."""
        pass

    def get_distance(self):
        """! Determine the detector distance based on the current geometry."""
        pass

    def integrate1d_pyfai(self):
        """! Perform 1D radial integration using pyFAI."""
        raise NotImplementedError

    def find_powder_pyfai(self,
                          expmt: str,
                          run: Union[str, int],
                          rootdir: str) -> np.ndarray:
        raise NotImplementedError

    def integrate1d_smd(self):
        """! Perform 1D radial integration using Small Data Tools."""
        raise NotImplementedError

    def find_powder_smd(self,
                        expmt: str,
                        run: Union[str, int],
                        rootdir: str) -> np.ndarray:
        raise NotImplementedError

    def plot_all(self):
        fig, axs = plt.subplots(2, 2, figsize=(6,6), dpi=150)
        q = self._saxsprofile.q_vals
        I = self._saxsprofile.intensity

        axs[0, 0] = plt.plot(q, I)
        axs[0, 0].set_xlabel(r'q (A$^{-1}$)')
        axs[0, 0].set_ylabel(r'I(q) (a.u.)')
        axs[0, 0].set_title(r'Scattering Profile')

        axs[0, 1] = plt.plot(q**2, np.log(I))
        axs[0, 1].set_xlabel(r'q$^{2}$ (A$^{-2}$)')
        axs[0, 1].set_ylabel(r'ln(I)')
        axs[0, 1].set_title('Guinier Plot')

        axs[1, 0] = plt.plot(q, I*q**4)
        axs[1, 0].set_xlabel(r'q (A$^{-1}$)')
        axs[1, 0].set_ylabel(r'Iq$^{4}$')
        axs[1, 0].set_title(r'Porod Plot')

        axs[1, 1] = plt.plot(q, I*q**2)
        axs[1, 1].set_xlabel(r'q (A$^{-1}$)')
        axs[1, 1].set_ylabel(r'Iq$^{2}$')
        axs[1, 1].set_title('Kratky Plot')

        fig.tight_layout()


    def plot_saxs_profile(self):
        """! Plot the SAXS profile and associated metrics. """
        pass

    @property
    def saxsprofile(self) -> SAXSProfile:
        """! Property to retrieve the SAXSProfile object.

        The SAXSProfile contains the q-values and associated intensities.
        Additional metadata, such as whether the sample is considered to be
        globular or rod-like is also contained within the object.
        """
        return self._saxsprofile

    @saxsprofile.setter
    def saxsprofile(self, val: SAXSProfile):
        self._saxsprofile = val

class GuinierAnalyzer:
    """! Class to perform Guinier Analysis of a SAXS profile."""
    def __init__(self, profile: SAXSProfile):
        self.profile: SAXSProfile = profile
        self.idx_qmin: int = 0
        self.idx_qmax: int = np.max(np.where(self.profile.q_vals < 0.04))

    def guinier_fit(self) -> GuinierResult:
        """ Perform a linear fit to extract Rg."""
        qvals = self.profile.q_vals
        intensities = self.profile.intensity

        # Transformations for fit
        x = (qvals[self.idx_qmin:self.idx_qmax+1])**2
        y = np.log(intensities[self.idx_qmin:self.idx_qmax+1])

        fit_result = linregress(x, y)

        I0 = np.exp(fit_result.slope)
        Rg = (-3*fit_result.intercept)**0.5

        guinier = GuinierResult(fit_result, I0, Rg)
        return guinier

    def determine_qmax(self):
        """ Iteratively perform the Guinier analysis and determine the best
        qmax.

        The product of the maximum q-value included in the analysis and the
        determined Radius of Gyration should be ~1.3 for globular structures
        or ~1 for extended/rod-shaped structures. This method iteratively
        performs the fitting procedure, updating which q-values to include in
        order to get as close as possible to these values.
        """
        if self.profile.globular:
            optimum_qmaxRg: float = 1.3
        else:
            optimum_qmaxRg: float = 1.0

        self.guinier = self.guinier_fit()
        qmax = self.profile.q_vals[self.idx_qmax]
        qmaxRg = qmax*self.guinier.Rg
        last_qmax = self.idx_qmax

        while qmaxRg != optimum_qmaxRg:
            # Would be infinite w/o break below
            if qmaxRg > optimum_qmaxRg:
                self.idx_qmax -= 1
            else:
                self.idx_qmax += 1
            self.guinier = self.guinier_fit()
            qmax = self.profile.q_vals[self.idx_qmax]
            qmaxRg = qmax*self.guinier.Rg

            if self.idx_qmax == last_qmax:
                break
            last_qmax = self.idx_qmax

    def plot_residuals(self, result):
        """ Plot the residuals from the fitting procedure.

        A "frown" indicates interparticle repulsion effects. A "smile" could
        mean aggregation.
        """
        qvals = self.profile.q_vals
        intensities = self.profile.intensity

        # Transformations for fit
        x = (qvals[self.idx_qmin:self.idx_qmax+1])**2
        y = np.log(intensities[self.idx_qmin:self.idx_qmax+1])

        line = x*result.slope + result.intercept

        residuals = y-line

        plt.plot(x, residuals)
        pass

class PorodAnalyzer:
    def __init__(self):
        pass

    def calc_surface_area(self):
        pass

class KratkyAnalyzer:
    pass
