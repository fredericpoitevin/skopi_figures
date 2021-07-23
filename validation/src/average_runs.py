import os, sys, glob, getopt
from psana import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import h5py as h5
from tqdm import tqdm
import logging

def config_info():
    experimentName='amo86615'
    detInfo='pnccdBack'
    return experimentName, detInfo

def main(argv):
    """
    """ 
    logging.basicConfig(filename='average_runs.log', 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                        level=logging.INFO)
    FIGURE_DIR="/cds/home/f/fpoitevi/Toolkit/skopi/examples/notebooks/figures"
    CXIDBlist_Filepath=FIGURE_DIR+"/pr772/singleHitList.h5"
    team="LCLS"
    output_dir=FIGURE_DIR+'/pr772/team_'+team+'/'

    logging.info("reading Hit List")
    singleHitList = h5.File(CXIDBlist_Filepath, 'r')
    run_list = np.unique(singleHitList[team]['runs'][:])
    logging.info("... list of runs found: {}".format(run_list))

    summed_image_total = None
    n_images_total = 0
    for run in run_list:
        logging.info('> summing run {}'.format(run))
        summed_image, n_images = average_run(run, 
                                             singleHitList[team],
                                             output_dir=output_dir)
        logging.info('   # of images {}'.format(n_images))
        n_images_total += n_images
        if summed_image_total is None:
            summed_image_total = summed_image
        else:
            summed_image_total += summed_image
    
    output_filename = 'team_{}'.format(team)
    np.save(output_filename, summed_image_total/n_images_total)
    logging.info('DONE! Wrote the sum of {} images to {}'.format(n_images_total, output_filename))

def average_run(runNumber, singleHits, output_dir="./"):
    """average_run
    """
    experimentName, detInfo = config_info()

    print("Run # {}".format(runNumber))
    print("retrieving background")
    background_img = retrieve_background_image()

    print("retrieving timestamps")
    timestamps, fiducials = retrieve_timestamps(runNumber, singleHits)
    print("# of hits: {}".format(len(timestamps)))

    print("retrieving images")
    img = retrieve_signal_image(runNumber,
                                timestamps,fiducials,
                                background_img,
                                output_dir)

    return img, len(timestamps)
    
def retrieve_signal_image(runNumber,
                           timestamps, fiducials,
                           background_img,
                           output_dir):
    """
    """
    experimentName, detInfo = config_info()
    try:
        ds = DataSource('exp='+experimentName+':run='+str(runNumber)+':idx')
        run = ds.runs().next()
        det = Detector(detInfo)
    except:
        raise Exception("data not found, skip")
    n_images = len(timestamps)
    batch_size=np.amin((n_images, 100))
    output_filename='{}/{}.npy'.format(output_dir,'r'+str(runNumber))
    n_batches=np.floor(n_images/batch_size).astype(int)
    print("... will go through {} batches".format(n_batches))
    for i_batch in tqdm(range(n_batches)):
        i_start = i_batch*batch_size
        i_end = np.amin((n_images, (i_batch+1)*batch_size))
        img = []
        for i in tqdm(range(i_start, i_end), leave=False):
            et = EventTime(int(timestamps[i]), fiducials[i])
            evt = run.event(et)
            image = det.image(evt)-background_img
            img.append(image)
        image = np.sum(np.array(img),axis=0)
        if(i_batch>0):
            image += np.load(output_filename)
        np.save(output_filename, image)
    image = np.load(output_filename)
    np.save(output_filename, image/n_images)
    print('DONE! Wrote {}'.format(output_filename))

    q_image = build_q_image(experimentName,runNumber)
    detector_mask = build_detector_mask(experimentName, runNumber)

    output_q_filename='{}/{}-q'.format(output_dir,'r'+str(runNumber))
    np.save(output_q_filename, q_image)
    print('DONE! Wrote {}'.format(output_q_filename))
    output_saxs_filename='{}/{}-saxs'.format(output_dir,'r'+str(runNumber))
    save_saxs(q_image, image/n_images, detector_mask, output_saxs_filename)
    print('DONE! Wrote {}'.format(output_saxs_filename))

    return image

def retrieve_timestamps(runNumber, singleHits):
    """
    """
    try:
        indices = np.where(singleHits['runs'][:]==runNumber)
        timestamps = singleHits['timestamps'][indices]
        fiducials  = singleHits['fiducials'][indices]
    except:
        raise Exception("empty run, skip")
    return timestamps, fiducials

def retrieve_background_image():
    """
    """
    experimentName, detInfo = config_info()
    ds = DataSource('exp='+experimentName+':run='+str(190)+':idx')
    run = ds.runs().next()
    det = Detector(detInfo)
    times = run.times()
    background_evt = run.event(times[4])
    background_img = det.image(background_evt)

    return background_img

def save_saxs(q_image, img_mean, detector_mask, output_file):
    """"""
    center = center_from_displacement(img_mean, ix=0, iy=0)
    
    profile_q = radial_profile(q_image, center, mask=detector_mask, threshold=0)
    profile_mean = radial_profile(img_mean, center, mask=detector_mask, threshold=0)
    profile_std = radial_profile(img_mean**2, center, mask=detector_mask, threshold=0)
    profile_std = np.sqrt(profile_std - profile_mean**2)
    profile_mask = np.where((profile_q > 0.00072) & (profile_q < 0.01), True, False)
    to_be_saved = np.stack((profile_q, profile_mean, profile_std, profile_mask)).T
    np.save(output_file, to_be_saved)

def radial_profile(data, center, mask=None, 
                   filter=False, filter_order=2, filter_threshold=0.25,
                   threshold=10):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    if mask is not None:
        r = np.where(mask==1, r, 0)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr   = np.bincount(r.ravel())
    radialprofile = np.divide(tbin, nr, out=np.zeros(nr.shape[0]), where=nr!=0)
    if filter:
        sos = butter(filter_order, filter_threshold, output='sos')
        radialprofile = sosfiltfilt(sos, radialprofile)
    radialprofile[radialprofile<threshold] = 0
    return radialprofile

def center_from_displacement(data, ix=0, iy=0):
    center = [(data.shape[0]+ix)/2,(data.shape[1]+iy)/2]
    #print(f'center = {center}')
    return center

def build_q_image(experimentName,run, itime=0):
    """
    """
    detInfo='pnccdBack'
    runNumber=np.uint8(run)
    ds = DataSource('exp='+experimentName+':run='+str(runNumber)+':idx')
    run = ds.runs().next()    
    times = run.times()
    evt = run.event(times[itime])
    env = ds.env()
    det = Detector(detInfo)
    img = det.image(evt)
    ix_center = img.shape[0]/2
    iy_center = img.shape[1]/2
    #
    pixelSize = det.pixel_size(evt)
    detectorDistance = np.abs(np.mean(det.coords_xyz(evt)[2]))
    wavelength = env.epicsStore().value('SIOC:SYS0:ML00:AO192')*10 # in Angstroem
    #
    ix, iy = det.indexes_xy(evt)
    indexes = np.sqrt((ix-ix_center)**2 + (iy-iy_center)**2)
    theta = np.arctan(indexes*pixelSize/detectorDistance)
    q = 2*np.sin(theta/2.)/wavelength
    #
    return det.image(evt=evt, nda_in=q)

def build_detector_mask(experimentName, run):
    detInfo='pnccdBack'
    runNumber=np.uint8(run)
    ds = DataSource('exp='+experimentName+':run='+str(runNumber)+':idx')
    run = ds.runs().next()
    times = run.times()
    evt = run.event(times[0])
    det = Detector(detInfo)
    return det.image(evt=evt, nda_in=det.mask_calib(evt))

### The next section are mostly obsolete function, for per-run analysis
def save_npy(CXIDBlist_Filepath, run, team='LCLS', outdir=None):
    experimentName, detInfo = config_info()
    #
    singleHitList=h5.File(CXIDBlist_Filepath, 'r')
    print(singleHitList.keys())
    runNumber=np.uint8(run)
    indices = np.where(singleHitList[team]['runs'][:]==runNumber)
    timestamps = singleHitList[team]['timestamps'][indices]
    fiducials  = singleHitList[team]['fiducials'][indices]
    
    ds = DataSource('exp='+experimentName+':run='+str(runNumber)+':idx')
    run = ds.runs().next()
    det = Detector(detInfo)
    
    # using psocake, Chuck identified this event as being background
    times = run.times()
    background_evt = run.event(times[4])
    background_img = det.image(background_evt)
    
    if outdir is not None:
        outdir += 'team_{}/{}/'.format(team,'r'+str(runNumber))
        if not os.path.exists(outdir):
            print('... creating {}'.format(outdir))
            os.makedirs(outdir)
        # save each image in a numpy array
        print("Saving team {} hits under: {}".format(team, outdir))
        idata=0
        for i in tqdm(range(len(timestamps))):
            et = EventTime(int(timestamps[i]), fiducials[i])
            evt = run.event(et)
            img = det.image(evt)
            img -= background_img
            sdata = '{0:04d}'.format(idata)
            np.save('{}{}.npy'.format(outdir,sdata), img)
            idata += 1

def average_npy(run, team='LCLS', outdir=None):
    runNumber=np.uint8(run)
    outdir += 'team_{}/{}/'.format(team,'r'+str(runNumber))
    if os.path.exists(outdir):
        filelist = glob.glob(outdir+'[!p]*.npy')
        n_images = len(filelist)
        print('Reading {} images from {}'.format(n_images, outdir))
        img_norms = np.zeros(n_images)
        for idata in tqdm(range(n_images)):
            sdata = '{0:04d}'.format(idata)
            img   = np.load('{}{}.npy'.format(outdir,sdata))
            img_norms[idata] = np.sum(img.flatten())
            if(idata==0):
                img_mean = img
                img_std  = img*img
            else:
                img_mean += img
                img_std  += img*img
        img_mean /= n_images
        img_std /= n_images
        img_std -= img_mean*img_mean
        img_std = np.sqrt(img_std)
        #
        return img_norms, img_mean, img_std
    
def summary(team, 
            runNumber, 
            experimentName='amo86615', 
            output_dir=None,
            vmin=1):
    # READ
    img_norms, img_mean, img_std = average_npy(runNumber, 
                                               team=team, outdir=output_dir)
    # PLOT
    plot_norms(img_norms, title='{}_r{}'.format(team, runNumber))
    cute_plot(img_mean, vmin=vmin, title='mean intensity')
    cute_plot(img_std, vmin=vmin,title='stdev intensity')
    # SAXS
    q_image = build_q_image(experimentName,runNumber)
    #cute_plot(q_image, log=False, title='q (Angstroem)')
    detector_mask = build_detector_mask(experimentName, runNumber)
    plot_saxs(q_image, img_mean, detector_mask, 
              output_dir+'team_{}/{}/'.format(team,'r'+str(runNumber)))

def cute_plot(data, vmin=1, figsize=4, dpi=180, title=None, log=True):
    fig = plt.figure(figsize=(figsize,figsize),dpi=dpi)
    img = np.ma.masked_where(data<=0,data)
    if log:
        plt.imshow(img, norm=LogNorm(vmin=vmin), interpolation='none')
    else:
        plt.imshow(img)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.xlabel('Y')
    plt.ylabel('X')
    
def plot_norms(img_norms, title=None):
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,
                                  figsize=(8,4),dpi=180, 
                                  sharey=True)
    if title is not None:
        fig.suptitle(title)
    ax1.plot(np.log10(img_norms))
    ax1.set_xlabel('image id')
    ax1.set_ylabel('image norm (log)')
    ax2.hist(np.log10(img_norms), orientation='horizontal', bins=100)
    ax2.set_xlabel('population')
    #plt.tight_layout()
    plt.show()

def plot_saxs(q_image, img_mean, detector_mask, output_dir):
    """"""
    center = center_from_displacement(img_mean, ix=0, iy=0)
    
    profile_q = radial_profile(q_image, center, mask=detector_mask, threshold=0)
    profile_mean = radial_profile(img_mean, center, mask=detector_mask, threshold=0)
    profile_std = radial_profile(img_mean**2, center, mask=detector_mask, threshold=0)
    profile_std = np.sqrt(profile_std - profile_mean**2)
    profile_mask = np.where((profile_q > 0.00072) & (profile_q < 0.01), True, False)

    fig = plt.figure(figsize=(4,4),dpi=180)
    plt.errorbar(profile_q[profile_mask], 
                 profile_mean[profile_mask], 
                 yerr=profile_std[profile_mask], 
                 elinewidth=0.1, color='black')
    plt.xlim(0)
    plt.grid()
    plt.yscale('log')
    #
    to_be_saved = np.stack((profile_q, profile_mean, profile_std, profile_mask)).T
    np.save('{}{}.npy'.format(output_dir,'pr772_1d_profile'), to_be_saved)

##################3
if __name__ == "__main__":
    main(sys.argv[1:])
