import numpy as np
import argparse
import h5py
import os
from sfx_utils.interfaces.psana_interface import *
from psalgos.pypsalgos import PyAlgos

class PeakFinder:
    
    """
    Perform adaptive peak-finding on a psana run and save the results to cxi
    format. Adapted from psocake.
    """
    
    def __init__(self, exp, run, det_type, outdir, tag='', mask=None, psana_mask=True, 
                 min_peaks=2, max_peaks=2048, npix_min=2, npix_max=30, amax_thr=80., 
                 atot_thr=120.,  son_min=7.0, peak_rank=3, r0=3.0, dr=2.0, nsigm=7.0):
        
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        # peak-finding algorithm parameters
        self.npix_min = npix_min # int, min number of pixels in peak
        self.npix_max = npix_max # int, max number of pixels in peak
        self.amax_thr = amax_thr 
        self.atot_thr = atot_thr
        self.son_min = son_min # in psocake, nsigm=son_min
        self.peak_rank = peak_rank # radius in which central pix is a local max, int
        self.r0 = r0 # radius in pixels of ring for background evaluation, float
        self.dr = dr # width in pixels of ring for background evaluation, float
        self.nsigm = nsigm # intensity threshold to include pixel in connected group, float
        self.min_peaks = min_peaks # int, min number of peaks per image
        self.max_peaks = max_peaks # int, max number of peaks per image
        
        # set up 
        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type, track_timestamps=True)
        self.psi.distribute_events(self.rank, self.size)
        self.n_events = self.psi.max_events
        self.set_up_cxi(outdir, tag)
        self.set_up_algorithm(mask_file=mask, psana_mask=psana_mask)  
        
    def _generate_mask(self, mask_file=None, psana_mask=True):
        """
        Generate mask, optionally a combination of the psana-generated mask
        and a user-supplied mask.
        
        Parameters
        ----------
        mask_file : str
            path to mask in shape of unassembled detector, optional
        psana_mask : bool
            if True, retrieve mask from psana Detector object
        """
        mask = np.ones(self.psi.det.shape()).astype(np.uint16)  
        if psana_mask:
            mask = self.psi.det.mask(self.psi.run, calib=False, status=True, 
                                     edges=False, centra=False, unbond=False, 
                                     unbondnbrs=False).astype(np.uint16)
        if mask_file is not None:
            mask *= np.load(mask_file).astype(np.uint16)
        
        self.mask = mask
        
    def set_up_algorithm(self, mask_file=None, psana_mask=True):
        """
        Set up the peak-finding algorithm. Currently only the adaptive
        variant is supported. For more details, see:
        https://github.com/lcls-psana/psalgos/blob/master/src/pypsalgos.py 
        """
        self._generate_mask(mask_file=mask_file, psana_mask=psana_mask)
        self.alg = PyAlgos(mask=self.mask, pbits=0) # pbits controls verbosity
        self.alg.set_peak_selection_pars(npix_min=self.npix_min,
                                         npix_max=self.npix_max,
                                         amax_thr=self.amax_thr,
                                         atot_thr=self.atot_thr,
                                         son_min=self.son_min)
        self.n_hits = 0
        self.powder_hits, self.powder_misses = None, None
        
        # additional self variables for tracking peak stats
        self.iX = self.psi.det.indexes_x(self.psi.run).astype(np.int64)
        self.iY = self.psi.det.indexes_y(self.psi.run).astype(np.int64)
        self.ipx, self.ipy = self.psi.det.point_indexes(self.psi.run, pxy_um=(0, 0))

    def set_up_cxi(self, outdir, tag=''):
        """
        Set up the CXI files to which peak finding results will be saved.
        
        Parameters
        ----------
        outdir : str
            output directory
        tag : str
            file nomenclature suffix, optional
        """
        if (tag != '') and (tag[0]!='_'):
            tag = '_' + tag
        self.fname = os.path.join(outdir, f'{self.psi.exp}_{self.psi.run:04}_{self.rank}{tag}.cxi')
        
        outh5 = h5py.File(self.fname, 'w')
        
        # entry_1 dataset for downstream processing with CrystFEL
        entry_1 = outh5.create_group("entry_1")
        keys = ['nPeaks', 'peakXPosRaw', 'peakYPosRaw', 'rcent', 'ccent', 'rmin',
                'rmax', 'cmin', 'cmax', 'peakTotalIntensity', 'peakMaxIntensity', 'peakRadius']
        ds_expId = entry_1.create_dataset("experimental_identifier", (self.n_events,), maxshape=(None,), dtype=int)
        ds_expId.attrs["axes"] = "experiment_identifier"
        
        # for storing images in crystFEL format
        det_shape = self.psi.det.shape()
        dim0, dim1 = det_shape[0] * det_shape[1], det_shape[2]
        data_1 = entry_1.create_dataset('/entry_1/data_1/data', (self.n_events, dim0, dim1), chunks=(1, dim0, dim1),
                                        maxshape=(None, dim0, dim1),dtype=np.float32)
        data_1.attrs["axes"] = "experiment_identifier"
        
        for key in ['powderHits', 'powderMisses']:
            entry_1.create_dataset(key, (dim0, dim1), chunks=(dim0, dim1), maxshape=(dim0, dim1), dtype=float)
                
        # peak-related keys
        for key in keys:
            if key == 'nPeaks':
                ds_x = outh5.create_dataset(f'/entry_1/result_1/{key}', (self.n_events,), maxshape=(None,), dtype=int)
                ds_x.attrs['minPeaks'] = self.min_peaks
                ds_x.attrs['maxPeaks'] = self.max_peaks
            else:
                ds_x = outh5.create_dataset(f'/entry_1/result_1/{key}', (self.n_events,self.max_peaks), 
                                            maxshape=(None,self.max_peaks), chunks=(1,self.max_peaks), dtype=float)
            ds_x.attrs["axes"] = "experiment_identifier:peaks"
            
        # LCLS dataset to track event timestamps
        lcls_1 = outh5.create_group("LCLS")
        keys = ['eventNumber', 'machineTime', 'machineTimeNanoSeconds', 'fiducial']
        
        for key in keys:
            ds_x = lcls_1.create_dataset(f'{key}', (self.n_events,), maxshape=(None,), dtype=int)
            ds_x.attrs["axes"] = "experiment_identifier"
            
        outh5.close()
    
    def store_event(self, outh5, img, peaks):
        """
        Store event's peaks in CXI file, converting to Cheetah conventions.
        
        Parameters
        ----------
        outh5 : h5py._hl.files.File
            open h5 file for storing output for this rank
        img : numpy.ndarray, shape (n_panels, n_panels_fs, n_panels_ss)
            calibrated detector data in shape of unassembled detector
        peaks : numpy.ndarray, shape (n_peaks, 17)
            results of peak finding algorithm for a single event
        """
        if self.psi.det_type not in ['jungfrau4M', 'epix10k2M']:
            print("Warning! Reformatting to Cheetah may not be correct")
        
        ch_rows = peaks[:,0] * img.shape[1] + peaks[:,1]
        ch_cols = peaks[:,2]
        
        # entry_1 entries for crystFEL processing
        outh5['/entry_1/data_1/data'][self.n_hits,:,:] = img.reshape(-1, img.shape[-1]) 
        outh5['/entry_1/result_1/nPeaks'][self.n_hits] = peaks.shape[0] 
        outh5['/entry_1/result_1/peakXPosRaw'][self.n_hits,:peaks.shape[0]] = ch_cols.astype('int')
        outh5['/entry_1/result_1/peakYPosRaw'][self.n_hits,:peaks.shape[0]] = ch_rows.astype('int')
        
        outh5['/entry_1/result_1/rcent'][self.n_hits,:peaks.shape[0]] = peaks[:,6] # row center of gravity
        outh5['/entry_1/result_1/ccent'][self.n_hits,:peaks.shape[0]] = peaks[:,7] # col center of gravity
        outh5['/entry_1/result_1/rmin'][self.n_hits,:peaks.shape[0]] = peaks[:,10] # minimal row of pixel group in the peak
        outh5['/entry_1/result_1/rmax'][self.n_hits,:peaks.shape[0]] = peaks[:,11] # maximal row of pixel group in the peak
        outh5['/entry_1/result_1/cmin'][self.n_hits,:peaks.shape[0]] = peaks[:,12] # minimal col pixel group in the peak
        outh5['/entry_1/result_1/cmax'][self.n_hits,:peaks.shape[0]] = peaks[:,13] # maximal col of pixel group in the peak
        
        outh5['/entry_1/result_1/peakTotalIntensity'][self.n_hits,:peaks.shape[0]] = peaks[:,5]
        outh5['/entry_1/result_1/peakMaxIntensity'][self.n_hits,:peaks.shape[0]] = peaks[:,4]
        outh5['/entry_1/result_1/peakRadius'][self.n_hits,:peaks.shape[0]] = self._compute_peak_radius(peaks)
        
        # LCLS dataset - currently omitting timetool information
        outh5['/LCLS/eventNumber'][self.n_hits] = self.psi.counter
        outh5['/LCLS/machineTime'][self.n_hits] = self.psi.seconds[-1]
        outh5['/LCLS/machineTimeNanoSeconds'][self.n_hits] = self.psi.nanoseconds[-1]
        outh5['/LCLS/fiducial'][self.n_hits] = self.psi.fiducials[-1]
        
    def curate_cxi(self):
        """
        Curate the CXI file by reshaping the keys and adding powders.
        """
        outh5 = h5py.File(self.fname,"r+")
        
        # add powders
        outh5["/entry_1/data_1/powderHits"][:] = self.powder_hits.reshape(-1, self.powder_hits.shape[-1])
        outh5["/entry_1/data_1/powderMisses"][:] = self.powder_misses.reshape(-1, self.powder_misses.shape[-1])       
        
        # resize the CrystFEL keys
        data_shape = outh5["/entry_1/data_1/data"].shape
        outh5['/entry_1/data_1/data'].resize((self.n_hits, data_shape[1], data_shape[2]))

        for key in ['peakXPosRaw', 'peakYPosRaw', 'rcent', 'ccent', 'rmin',
                    'rmax', 'cmin', 'cmax' 'peakTotalIntensity', 'peakMaxIntensity', 'peakRadius']:
            outh5[f'/entry_1/result_1/{key}'].resize(self.n_hits, self.max_peaks)
            
        # crop the LCLS keys
        for key in ['eventNumber', 'machineTime', 'machineTimeNanoSeconds', 'fiducial']:
            outh5[f'/LCLS.{key}'].resize((self.n_events,))
            
        outh5.close()
    
    def _compute_peak_radius(self, peaks):
        """
        Compute radii of peaks based on their constituent pixels.
        
        Parameters
        ----------
        peaks : numpy.ndarray, shape (n_peaks, 17)
            results of peak finding algorithm for a single event
            
        Returns 
        -------
        radius : numpy.ndarray, shape (n_peaks)
            radii of peaks in pixels
        """
        cenX = self.iX[np.array(peaks[:, 0], dtype=np.int64),
                       np.array(peaks[:, 1], dtype=np.int64),
                       np.array(peaks[:, 2], dtype=np.int64)] + 0.5 - self.ipx
        cenY = self.iY[np.array(peaks[:, 0], dtype=np.int64),
                       np.array(peaks[:, 1], dtype=np.int64),
                       np.array(peaks[:, 2], dtype=np.int64)] + 0.5 - self.ipy
        return np.sqrt((cenX ** 2) + (cenY ** 2))
        
    def find_peaks_event(self, img):
        """
        Find peaks on a single image.
        
        Parameters
        ----------
        img : numpy.ndarray, shape (n_panels, n_panels_fs, n_panels_ss)
            calibrated detector data in shape of unassembled detector
            
        Returns
        -------
        peaks : numpy.ndarray, shape (n_peaks, 17)
            results of peak finding algorithm for this image
        """
        peaks = self.alg.peak_finder_v3r3(img, rank=self.peak_rank, 
                                          r0=self.r0, dr=self.dr, nsigm=self.nsigm) 
        return peaks
    
    def find_peaks(self):
        """
        Find all peaks in the images assigned to this rank.
        """
        
        start_idx, end_idx = self.psi.counter, self.psi.max_events
        outh5 = h5py.File(self.fname,"r+")

        for idx in range(start_idx, end_idx):
            # retrieve calibrated image
            evt = self.psi.runner.event(self.psi.times[self.psi.counter])
            self.psi.get_timestamp(evt.get(EventId))
            img = self.psi.det.calib(evt=evt)
            
            # search for peaks and store if found
            peaks = self.find_peaks_event(img)
            if (peaks.shape[0] >= self.min_peaks) and (peaks.shape[0] <= self.max_peaks):
                self.store_event(outh5, img, peaks)
                self.n_hits+=1
                print(idx, peaks.shape[0], self.psi.fiducials[-1])
                
            # generate / update powders
            if peaks.shape[0] >= self.min_peaks:
                if self.powder_hits is None:
                    self.powder_hits = img
                else:
                    self.powder_hits = np.maximum(self.powder_hits, img)
            else:
                if self.powder_misses is None:
                    self.powder_misses = img
                else:
                    self.powder_misses = np.maximum(self.powder_misses, img)
                
            self.psi.counter+=1
            
        outh5.close()
                        
#### For command line use ####
            
def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', help='Experiment name', required=True, type=str)
    parser.add_argument('-r', '--run', help='Run number', required=True, type=int)
    parser.add_argument('-d', '--det_type', help='Detector name, e.g epix10k2M or jungfrau4M',  required=True, type=str)
    parser.add_argument('-o', '--outdir', help='Output directory for cxi files', required=True, type=str)
    parser.add_argument('-t', '--tag', help='Tag to append to cxi file names', required=False, type=str, default='')
    parser.add_argument('-m', '--mask', help='Binary mask', required=False, type=str)
    parser.add_argument('--psana_mask', help='If True, apply mask from psana Detector object', required=False, type=bool, default=True)
    parser.add_argument('--min_peaks', help='Minimum number of peaks per image', required=False, type=int, default=2)
    parser.add_argument('--max_peaks', help='Maximum number of peaks per image', required=False, type=int, default=2048)
    parser.add_argument('--npix_min', help='Minimum number of pixels per peak', required=False, type=int, default=2)
    parser.add_argument('--npix_max', help='Maximum number of pixels per peak', required=False, type=int, default=30)
    parser.add_argument('--amax_thr', help='', required=False, type=float, default=80.)
    parser.add_argument('--atot_thr', help='', required=False, type=float, default=120.)
    parser.add_argument('--son_min', help='', required=False, type=float, default=7.0)
    parser.add_argument('--peak_rank', help='Radius in which central peak pixel is a local maximum', required=False, type=int, default=3)
    parser.add_argument('--r0', help='Radius of ring for background evaluation in pixels', required=False, type=float, default=3.0)
    parser.add_argument('--dr', help='Width of ring for background evaluation in pixels', required=False, type=float, default=2.0)
    parser.add_argument('--nsigm', help='Intensity threshold to include pixel in connected group', required=False, type=float, default=7.0)
    
    return parser.parse_args()

if __name__ == '__main__':
    
    params = parse_input()
    pf = PeakFinder(exp=params.exp, run=params.run, det_type=params.det_type, outdir=params.outdir, tag=params.tag,
                    mask=params.mask, psana_mask=params.psana_mask, min_peaks=params.min_peaks, max_peaks=params.max_peaks,
                    npix_min=params.npix_min, npix_max=params.npix_max, amax_thr=params.amax_thr, atot_thr=params.atot_thr, 
                    son_min=params.son_min, peak_rank=params.peak_rank, r0=params.r0, dr=params.dr, nsigm=params.nsigm)
    pf.find_peaks()
