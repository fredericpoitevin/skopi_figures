import numpy as np
from matplotlib import pyplot as plt
from average_runs import *

summed_image_file = "team_LCLS.npy"
output_saxs_file = "team_LCLS_saxs.npy"
output_saxs_png  = "team_LCLS_saxs.png"

image = np.load(summed_image_file)
print("Load summed image of shape: {}".format(image.shape))

center = center_from_displacement(image)
print("Image center: {}".format(center))

experimentName, detInfo = config_info()

print("Computing radial profile...")
q_image = build_q_image(experimentName, 190)
detector_mask = build_detector_mask(experimentName, 190)
profile_q = radial_profile(q_image, center, mask=detector_mask, threshold=0)
profile_mean = radial_profile(image, center, mask=detector_mask, threshold=0)
profile_std = radial_profile(image**2, center, mask=detector_mask, threshold=0)
profile_std = np.sqrt(profile_std - profile_mean**2)
profile_mask = np.where((profile_q > 0.00072) & (profile_q < 0.01), True, False)

print("Saving to file: {}".format(output_saxs_file))
saxs_profile = np.stack((profile_q, profile_mean, profile_std, profile_mask)).T
np.save('{}'.format(output_saxs_file), saxs_profile)

print("Plotting to file: {}".format(output_saxs_png))
fig = plt.figure(figsize=(4,4),dpi=180)
plt.title("AMO86615 | PR772 radial profile")
plt.ylabel('ADU')
plt.xlabel('s = 2.sin(theta)/lambda (inverse Angstroem)')
plt.errorbar(profile_q[profile_mask], profile_mean[profile_mask], yerr=profile_std[profile_mask], 
             elinewidth=0.1, color='black')
plt.xlim(0)
plt.grid()
plt.yscale('log')
plt.savefig(output_saxs_png)
