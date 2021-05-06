import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import skopi as sk
import h5py, time, os
import argparse
from mpi4py import MPI


def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description="SPI benchmarking for one rank.")
    parser.add_argument('-b', '--beam_file', help='Beam file', required=True, type=str)
    parser.add_argument('-p', '--pdb_file', help='Pdb file', required=True, type=str)
    parser.add_argument('-d', '--det_info', help='Detector info. Either (n_pixels, length, distance) for SimpleSquare'+
                        'or (det_type, geom_file, distance) for LCLSDetectors. det_type could be pnccd, for instance',
                        required=True, nargs=3)
    parser.add_argument('-e', '--experiment', help='SPI, SPI_agg, or FXS', required=True, type=str)
    parser.add_argument('-n', '--n_images', help='Number of slices to compute', required=True, type=int)
    parser.add_argument('-o', '--out_dir', help='Path to h5 output file', required=True, type=str)
    parser.add_argument('-r', '--ref_xyz_file', help='xyz coordinates for holography reference', required=False, type=str)

    return vars(parser.parse_args())


def configure_detector(det_info):
    """
    Configure detector. 
    
    :param det_info: string of detector information, either
        (n_pixels,length,distance) for SimpleSquare, or
        (det_type, geom_file, distance) for LCLSDetectors.
    :return det: detector object.
    """
    if det_info[0].isdigit():
        n_pixels, det_size, det_dist = det_info
        det = sk.SimpleSquareDetector(int(n_pixels), float(det_size), float(det_dist)) 
    elif det_info[0] == 'pnccd':
        det = sk.PnccdDetector(geom=det_info[1])
        det.distance = float(det_info[2])
    elif det_info[0] == 'cspad':
        det = sk.CsPadDetector(geom=det_info[1])
        det.distance = float(det_info[2])
    elif det_info[0] == 'jungfrau':
        det = sk.JungfrauDetector(geom=det_info[1], cameraConfig="fixedMedium")
        det.distance = float(det_info[2])
    elif args['det_info'][0] == 'epix10k':
        det = sk.Epix10kDetector(geom=det_info[1], cameraConfig="fixedMedium")
        det.distance = float(det_info[2])
    else:
        print("Detector type not recognized.")
        return
        
    return det


def setup_experiment(args, increase_factor=1e2):
    """
    Set up experiment class.
    
    :param args: dict containing beam, pdb, and detector info
    :param increase_factor: factor by which to increase beam fluence
    :return exp: SPIExperiment object
    """
    
    beam = sk.Beam(args['beam_file'])
    if increase_factor != 1:
        beam.set_photons_per_pulse(increase_factor*beam.get_photons_per_pulse())
    
    particle = sk.Particle()
    particle.read_pdb(args['pdb_file'], ff='WK')

    det = configure_detector(args['det_info'])
    jet_radius = 1e-6
    
    if args['experiment'] == 'SPI':
        exp = sk.SPIExperiment(det, beam, particle)
    elif args['experiment'] == 'SPI_agg':
        exp = sk.SPIExperiment(det, beam, particle, n_part_per_shot=2)
    elif args['experiment'] == 'FXS':
        exp = sk.FXSExperiment(det, beam, jet_radius, [particle], n_part_per_shot=2)
    elif args['experiment'] == 'holography':
        xyz = np.loadtxt(args['ref_xyz_file'])
        rparticle = sk.Particle()
        rparticle.create_from_atoms([("AU", xyz[i]) for i in range(xyz.shape[0])])
        exp = sk.HOLOExperiment(det, beam, [rparticle], [particle], jet_radius=jet_radius, ref_jet_radius=jet_radius)
    else:        
        print("Experiment type not recognized")
        return

    return exp


def simulate(args):
    """
    Simulate diffraction images, dividing the computation between ranks, and assemble
    the individual h5 files output by each rank into a single virtual dataset.
    
    :param args: dictionary of command line input
    """
    print("Simulating diffraction images")
    
    # set up MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    # determining number of images per rank
    n_batch, remainder = np.divmod(args['n_images'], size)
    if remainder != 0:
        print(f"Warning: number of requested images not divisible by number of ranks")
        print(f"Reducing images per rank to {n_batch}")
        args['n_images'] = size * n_batch
    
    # set up experiment 
    exp = setup_experiment(args)
    
    # simulate images and save to h5 file
    start_time = time.time()
    f = h5py.File(os.path.join(args["out_dir"], f"simulated_{rank}.h5"), "w")
    f.create_dataset("pixel_position_reciprocal", data=exp.det.pixel_position_reciprocal) # s-vectors in m-1 
    f.create_dataset("pixel_index_map", data=exp.det.pixel_index_map) # indexing map for reassembly

    photons = f.create_dataset("photons", shape=((n_batch,) + exp.det.shape))
    intensities = f.create_dataset("intensities", shape=((n_batch,) + exp.det.shape))
    orientations = f.create_dataset("orientations", shape=(n_batch, 4))
    
    for num in range(n_batch):
        results = exp.generate_image_stack(return_intensities=True, return_photons=True, return_orientations=True)
        photons[num,:,:], intensities[num,:,:], orientations[num] = results[0][0], results[0][1], results[1]
    f.close()
    
    return


def main():

    args = parse_input()
    if not os.path.isdir(args['out_dir']):
        os.mkdir(args['out_dir'])

    simulate(args)


if __name__ == '__main__':
    main()
