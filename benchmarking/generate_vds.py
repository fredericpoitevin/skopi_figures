import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py, time, os
import argparse, glob


def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description="Generate a virtual dataset from constituent h5 files.")
    parser.add_argument('-i', '--input_dir', help='Path to a collection of h5 files', required=True, type=str)
    parser.add_argument('-o', '--out_file', help='Output name', required=False, type=str)

    return vars(parser.parse_args())


def assemble_image_stack_batch(image_stack, index_map):
    """
    Assemble the image stack to obtain a 2D pattern according to the index map.
    Modified from skopi.
    :param image_stack: [stack num, panel num, panel pixel num x, panel pixel num y]
    :param index_map: [panel num, panel pixel num x, panel pixel num y]
    :return: [stack num, 2d pattern x, 2d pattern y]
    """
    # get boundary
    index_max_x = np.max(index_map[:, :, :, 0]) + 1
    index_max_y = np.max(index_map[:, :, :, 1]) + 1
    # get stack number and panel number
    stack_num = image_stack.shape[0]
    panel_num = image_stack.shape[1]

    # set holder
    image = np.zeros((stack_num, index_max_x, index_max_y))

    # loop through the panels
    for l in range(panel_num):
        image[:, index_map[l, :, :, 0], index_map[l, :, :, 1]] = image_stack[:, l, :, :]

    return image


def display_selection(input_dir, out_file, fnames, data_shapes):
    """
    Display a few images from the virtual dataset and confirm that they match
    their correpsonding images from the constituent files.

    :param input_dir: path to constituent h5 files
    :param out_file: virtual dataset file 
    :param fnames: list of constituent file names
    :param data_shapes: dictionary of constituent data keys and their shapes
    """
    # set up
    key = 'intensities'
    images_vds = np.zeros((3,) + data_shapes[key][1:])
    images_ind = np.zeros((3,) + data_shapes[key][1:])
    n_batch, n_total = data_shapes[key][0], data_shapes[key][0] * len(fnames)
    rand_ind = np.random.randint(0, high=n_total, size=3)

    # retrieve virtual dataset images
    with h5py.File(out_file, "r") as f:
        pixel_index_map = f['pixel_index_map'][:]
        for num in range(3):
            images_vds[num] = f['intensities'][rand_ind[num]]
    
    # retrieve constituent file images
    for num in range(3):
        quot, remainder = np.divmod(rand_ind[num], n_batch)
        with h5py.File(fnames[quot], "r") as f:
            images_ind[num] = f['intensities'][remainder]

    # plot select images
    f, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, figsize=(9,6))
    
    images_ind = assemble_image_stack_batch(images_ind, pixel_index_map)
    images_vds = assemble_image_stack_batch(images_vds, pixel_index_map)

    for num,ax in enumerate([ax1,ax2,ax3]):
        ax.imshow(images_ind[num], vmax=3*images_ind[num].mean())
    for num,ax in enumerate([ax4,ax5,ax6]):
        ax.imshow(images_vds[num], vmax=3*images_ind[num].mean())
    
    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        ax.set_xticks([])
        ax.set_yticks([])
    
    ax1.set_ylabel("Constituent image", fontsize=12)
    ax4.set_ylabel("Virtual dataset image", fontsize=12)

    f.savefig(f"{input_dir}/check.png", dpi=300, bbox_inches='tight')

    return


def generate_vds(input_dir, out_file):
    """
    Generate a virtual dataset from all the .h5 files present in input_dir.

    :param input_dir: directory containing input files
    :param out_file: path to output virtual dataset file
    """
    # retrieve constituent file names and data shapes
    fnames = glob.glob(f"{input_dir}/*.h5")
    data_shapes = dict()
    with h5py.File(fnames[0], "r") as f:
        for key in f.keys():
            data_shapes[key] = f[key].shape

    # add orientations
    key = 'orientations'
    n_batch, n_total = data_shapes[key][0], data_shapes[key][0] * len(fnames)
    layout = h5py.VirtualLayout(shape=(n_total,4), dtype=float)
    for n in range(len(fnames)):
        vsource = h5py.VirtualSource(fnames[n], key, shape=data_shapes[key])
        layout[n_batch*n:n_batch*(n+1)] = vsource
    with h5py.File(out_file, "w", libver="latest") as f:
        f.create_virtual_dataset(key, layout, fillvalue=-1)
        
    # add images from constituent files
    for key,dt in zip(['photons', 'intensities'], [int,float]):
        n_batch, n_total = data_shapes[key][0], data_shapes[key][0] * len(fnames)
        layout = h5py.VirtualLayout(shape=((n_total,)+data_shapes[key][1:]), dtype=dt)
        for n in range(len(fnames)):
            vsource = h5py.VirtualSource(fnames[n], key, shape=data_shapes[key])
            layout[n_batch*n:n_batch*(n+1)] = vsource
        with h5py.File(out_file, "a", libver="latest") as f:
            f.create_virtual_dataset(key, layout, fillvalue=-1)

    # add keys common to all files
    for key,dt in zip(['pixel_index_map', 'pixel_position_reciprocal'], [int,float]):
        layout = h5py.VirtualLayout(shape=data_shapes[key], dtype=dt)
        for n in range(len(fnames)):
            vsource = h5py.VirtualSource(fnames[n], key, shape=data_shapes[key])
            layout[:] = vsource
        with h5py.File(out_file, "a", libver="latest") as f:
            f.create_virtual_dataset(key, layout, fillvalue=-1)

    # check that vds generation worked
    display_selection(input_dir, out_file, fnames, data_shapes)

    return


def main():

    args = parse_input()
    if args['out_file'] is None:
        args['out_file'] = os.path.join(args['input_dir'], "simulated_vds.h5")

    generate_vds(args['input_dir'], args['out_file'])


if __name__ == '__main__':
    main()
