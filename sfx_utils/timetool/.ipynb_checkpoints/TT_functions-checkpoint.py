import numpy as np
import scipy as sp
import os,re,sys,optparse,glob,time,datetime,collections, socket
import os.path as osp
import TimeTool as TT
import psana
import matplotlib as mpl
import matplotlib.pyplot as plt
from psana import *

def relative_time(edge_pos,a,b,c):
    """
    Translate edge position into fs (for cxij8816)
     
    from docs >> fs_result = a + b*x + c*x^2, x is edge position
    """
    x = edge_pos
    tt_correction = a + b*x + c*x**2
    return  tt_correction

def rel_time(edge_pos,model):
    """
    Translate edge position into time
     
    from docs >> fs_result = a + b*x + c*x^2, x is edge position
    """
    if len(model) == 2:
        a = model[1]
        b = model[0]
        c = 0
    elif len(model) ==3:
        a = model[2]
        b = model[1] 
        c = model[0]
    x = edge_pos
    tt_correction = a + b*x + c*x**2
    return  tt_correction


def absolute_time(rel_time, nom_time):
    """
    Calculate actual delay from nominal time and TT correction
     
    """

    delay = (nom_time + rel_time)*1e06
    return delay

def TTcalib(roi, calib_run, exp, make_plot=False, poly=2):
    """
    Calibration of time tool:
    roi = region of interest on detector that is being used to determine the edge
    calib_run = run number of the calibration run
    exp = experiment number including instrument (e.g. 'cxilz0720' for run LZ0720 at CXI)
    make_plot: if true automatically plots the calibration curve and fit
    poly = polynomial used for fitting calibration curve, 1 or 2, default 2 as in confluence documentation
    returns model that can be used to determine the delay using the function 'rel_time'
    """

    ttOptions = TT.TimeTool.AnalyzeOptions(get_key='Timetool', eventcode_nobeam=13, sig_roi_y='30 50')
    ttAnalyze = TT.TimeTool.PyAnalyze(ttOptions)
    analyze = TT.TimeTool.PyAnalyze(ttOptions)
    ds = psana.DataSource(f'exp={exp}:run={calib_run}', module=ttAnalyze)
    
    edge_pos = []
    amp = []
    time = []

    for idx,evt in enumerate(ds.events()):
        ttdata = ttAnalyze.process(evt)
        if ttdata is None: continue
        edge_pos = np.append(edge_pos, ttdata.position_pixel())
        amp = np.append(amp,ttdata.amplitude())
        time = np.append(time,ds.env().epicsStore().value('LAS:FS5:VIT:FS_TGT_TIME_DIAL'))

    model = polyfit(edge_pos, time, poly)

    if make_plot:
        if poly = 1:
            model_time = model[0]*edge_pos+model[1]
        elif poly = 2:
            model_time = model[0]**2*edge_pos + model[1]*edge_pos + model[2]
        else:
            print('polynomial not defined, use 1st or 2nd order')
            break
   
        plt.plot(edge_pos,time, 'o', color='black',label='edge position')
        plt.plot(edge_pos, model_time, color='red',label = 'calibration fit')
        plt.xlabel('pixel edge')
        plt.ylabel('laser delay')
        plt.legend()

    return model


def get_delay(run_start, run_end, roi='30 50', expID, outDir, calib_model=[])
    """
    Function to determine the delay using the time tool:
    run_start, run_end: first and last run to analyze
    roi = region of interest on detector that is being used to determine the edge
    expID = experiment number including instrument (e.g. 'cxilz0720' for run LZ0720 at CXI)
    outDir = directory where output should be saved
    calib_model = output from time tool calibration (using 'TTcalib'), if not empty TTanalysis is performed again (direct = False)

    saves .txt files linking a delay time to each shot, identified by a stamp 
    each row in the output file: ['644172952-167590310-79638','-1275.255309579068']
    """    

    ttOptions = TT.AnalyzeOptions(get_key='Timetool', eventcode_nobeam=13, sig_roi_y=roi)
    ttAnalyze = TT.PyAnalyze(ttOptions)
    
    if len(calib_model) == 0:
        direct = True
    else:
        direct = False

    runs = np.arange(run_start,run_end+1)
    for run_number in runs:
        ds = psana.DataSource('exp=cxilz0720:run=' + str(run_number), module=ttAnalyze)
        evr_det = psana.Detector('evr1')
        edge_pos = []
        amp = []
        time = []
        evt = []
        stamp = []
        tt_delay = []
        abs_delay = []

        if direct = False:
            for idx,evt in enumerate(ds.events()):
                ec = evr_det.eventCodes(evt)
                if ec is None: continue
                ttdata = ttAnalyze.process(evt)
                if ttdata is None: continue
                eid = evt.get(EventId)
                fid = eid.fiducials()
                sec = eid.time()[0]
                stamp = np.append(stamp, str(sec) + "-" + str(nsec) + "-" + str(fid))
                edge_pos = np.append(edge_pos, ttdata.position_pixel())
                time = time = np.append(time, ds.env().epicsStore().value('LAS:FS5:VIT:FS_TGT_TIME_DIAL'))
        if direct = True:
            for idx,evt in enumerate(ds.events()):
                ec = evr_det.eventCodes(evt)
                if ec is None: continue
                eid = evt.get(EventId)
                fid = eid.fiducials()
                sec = eid.time()[0]
                stamp = np.append(stamp, str(sec) + "-" + str(nsec) + "-" + str(fid))
                edge_pos = np.append(edge_pos,ds.env().epicsStore().value('CXI:TIMETOOL:FLTPOS'))
                time = time = np.append(time, ds.env().epicsStore().value('LAS:FS5:VIT:FS_TGT_TIME_DIAL'))
        tt_delay = rel_time(edge_pos, calib_model)
        abs_delay = absolute_time(time, tt_delay)

        output = np.column_stack([stamp, abs_delay])
        fn = 'outDir' + str(run_number) + '.txt'
        fOn = np.savetxt(fn, output, delimiter=',', fmt = '%s')  
        
