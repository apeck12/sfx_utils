from psana import *

run=33
tag='_null'
ds =  MPIDataSource(f'exp=cxix53120:run={run}:dir=/cds/data/drpsrcf/cxi/cxix53120/xtc')
det = Detector('jungfrau4M')
smldata = ds.small_data((f'run{run}{tag}.h5',gather_interval=100)

for nevt,evt in enumerate(ds.events()):
   calib = det.calib(evt)
   if calib is None: continue
   mask = det.status(evt)[0]>0
   det_img = ~mask*calib
   det_manually_masked = det_img[(det_img<1e4) & (det_img >=-3)]
   det_sum = det_manually_masked.sum()      # number
   smldata.event(det_sum=det_sum)
