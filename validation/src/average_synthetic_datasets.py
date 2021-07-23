from average_runs import *
from psana import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import h5py as h5
from tqdm import tqdm
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main():
    """
    """
    logging.basicConfig(filename='average_synthetic_datasets_5.log',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    FIGURE_DIR="/cds/home/f/fpoitevi/Toolkit/skopi/examples/notebooks/figures/pr772/"
    SKOPI_DATASETS_DIR="/reg/data/ana03/scratch/fpoitevi/skopi/6q5u/"
    #density_list = ['0.325', '0.35', '0.375', '0.4', '0.425'] #['0.', '0.15', '0.3', '0.45', '0.6', '0.75']
    meshsize_list = ['2.00', '1.00']
    #for density in density_list:
    density='0.4'
    for meshsize in meshsize_list:
        filepath=SKOPI_DATASETS_DIR+density+"/"+meshsize+"/saveHDF5_parallel.h5"
        dataset = h5.File(filepath, 'r')

        logging.info("Processing {}".format(filepath))
        img_mean, img_std, n_images = reduce_dataset_images(dataset)
        
        logging.info("... Assembling the images reduced from {} images".format(n_images))
        img_mean_assembled = assemble_image(img_mean)
        img_std_assembled = assemble_image(img_std)

        output_filename=FIGURE_DIR+"skopi/6q5u_{}-{}-mean.npy".format(density, meshsize)
        logging.info("... Saving mean in {}".format(output_filename))
        np.save(output_filename, img_mean_assembled)
        
        output_filename=FIGURE_DIR+"skopi/6q5u_{}-{}std.npy".format(density, meshsize)
        logging.info("... Saving std in {}".format(output_filename))
        np.save(output_filename, img_std_assembled)
        
        center = center_from_displacement(img_mean_assembled)
        experimentName, detInfo = config_info()

        logging.info("... Computing radial profile...")
        image = img_mean_assembled
        q_image = build_q_image(experimentName, 190)
        detector_mask = build_detector_mask(experimentName, 190)
        profile_q = radial_profile(q_image, center, mask=detector_mask, threshold=0)
        profile_mean = radial_profile(image, center, mask=detector_mask, threshold=0)
        profile_std = radial_profile(image**2, center, mask=detector_mask, threshold=0)
        profile_std = np.sqrt(profile_std - profile_mean**2)
        profile_mask = np.where((profile_q > 0.000) & (profile_q < 10), True, False)

        output_saxs_file = FIGURE_DIR+"skopi/6q5u_{}-{}-saxs.npy".format(density, meshsize)
        logging.info("...Saving to file: {}".format(output_saxs_file))
        saxs_profile = np.stack((profile_q, profile_mean, profile_std, profile_mask)).T
        np.save('{}'.format(output_saxs_file), saxs_profile)

        output_saxs_png = FIGURE_DIR+"skopi/6q5u_{}-{}-saxs.png".format(density, meshsize)
        logging.info("... Plotting to file: {}".format(output_saxs_png))
        fig = plt.figure(figsize=(4,4),dpi=180)
        plt.title("SKOPI | PR772 with inner density = {} e/A**3".format(density))
        plt.ylabel('ADU')
        plt.xlabel('s = 2.sin(theta)/lambda (inverse Angstroem)')
        plt.errorbar(profile_q[profile_mask], profile_mean[profile_mask], yerr=profile_std[profile_mask],
                     elinewidth=0.1, color='black')
        plt.xlim(0)
        plt.grid()
        plt.yscale('log')
        plt.savefig(output_saxs_png)

def plot_central_slices(file, title=None):
    """
    """
    dataset = h5.File(file, 'r')
    volume = np.asarray(dataset['volume'])
    center_slice = int(np.floor(volume.shape[0]/2))
    
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9,3), dpi=200)
    
    if title is not None:
        plt.suptitle(title)
        
    ax=axs[0]
    ax.set_title('XY slice')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    x = ax.imshow(np.abs(volume[:,:,center_slice]).T, norm=LogNorm(), origin='lower')
    fig.colorbar(x, cax=cax, orientation='vertical')
    
    ax=axs[1]
    ax.set_title('XZ slice')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    x = ax.imshow(np.abs(volume[:,center_slice,:]).T, norm=LogNorm(), origin='lower')
    fig.colorbar(x, cax=cax, orientation='vertical')
    
    ax = axs[2]
    ax.set_title('YZ slice')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    x = ax.imshow(np.abs(volume[center_slice, :,:]).T, norm=LogNorm(), origin='lower')
    fig.colorbar(x, cax=cax, orientation='vertical')
    
    plt.tight_layout()
    plt.show()
    
def assemble_image(panels):
    """
    """
    experimentName, detInfo = config_info()
    runNumber = '190'
    ds = DataSource('exp='+experimentName+':run='+runNumber+':idx')
    run = ds.runs().next()    
    times = run.times()
    evt = run.event(times[0])
    #env = ds.env()
    det = Detector(detInfo)
    return det.image(evt=evt, nda_in=panels)

def reduce_dataset_images(dataset, n_images=None):
    """
    """
    photons = dataset['photons']
    if n_images is None:
        n_images = photons.shape[0]
    #print(n_images)
    for idata in tqdm(range(n_images)):
        img = photons[idata]
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
    return img_mean, img_std, n_images

##########
if __name__ == "__main__":
    main()
