# Validation

## # PR772 SAXS profile - comparison between `skopi` and experiment

### Generate synthetic SAXS profile with `skopi`

### Generate SAXS profile from AMO86615 experiment

See `pr772_make_1d_profile.ipynb` notebook for some information on how the hits identified in the data of experiment AMO86615 were summed and radially averaged to make the "experimental SAXS profile". The actual scripts and results can be found under `pr772/`.

####  Get list of hits

Three teams have deposited their analysis on CXIDB at the following link: [id-58](https://www.cxidb.org/id-58.html).
We downloaded the file named `singleHitList.h5` and used it to retrieve the corresponding images.

#### Sum diffraction patterns of all hits

We ran the following script on `psanagpu` servers at LCLS, with access to the amo86615 experiment (To check that the images are "on disk", you can do: `./list_data.sh`). The corresponding logfile is `average_runs.log`.

```bash
(ana-4.0.17) [user@psanagpu pr772/]$ python average_runs.py
```

The resulting summed image is `team_LCLS.npy` which can be simply visualized with:
```python
img_mean = np.load("team_LCLS.npy")
fig = plt.figure(figsize=(9,9),dpi=360)
plt.title('AMO88615 | PR772 hits summed')
img = np.where(img_mean<1e-6, 1e-6, img_mean)
plt.imshow(img.T, origin='lower', norm=LogNorm(vmin=1e-2), interpolation='gaussian')
plt.colorbar(label='ADU')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("team_LCLS.png")
```
<img src="https://github.com/fredericpoitevin/skopi/blob/main/examples/notebooks/figures/pr772/team_LCLS.png">

### Radially average to get the SAXS profile

Running the following script will generate `team_LCLS_saxs.npy` and the corresponding `team_LCLS_saxs.png` figure displayed below.

```bash
(ana-4.0.17) [user@psanagpu pr772/]$ python make_saxs_profile.py
```
<img src="https://github.com/fredericpoitevin/skopi/blob/main/examples/notebooks/figures/pr772/team_LCLS_saxs.png">


