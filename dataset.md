* Approx. 22000 subjects
* preprocessed using [standard pipelines](https://biobank.ctsu.ox.ac.uk/crystal/docs/brain_mri.pdf)
* standard wavelet denoising
* freesurfer structural pre-processing
* not in BIDS format

## Structural data
* `anat` subdir
* T1
  * FSL [FAST](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FAST) and [FIRST](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIRST) outout
* T2 FLAIR imaging
  * automated lesion masking
* `unbiased_brain.nii.gz` with corrected bias field

Freesurfer reconstruction
* 19891 datasets in `~/anat/surfaces`
  
Structural parcellation
* desikan killiany atlas
* 308 parcellation
* glasser parcellation

## Functional data
* rs-fMRI
* first-stage timeseries in ICA analysis
* denoising
* wavelet despiking method

Functional parcellation
* aparc
* 308
* HCP (360 regions)
* SJH (1012 regions)


Machine learning classification 
* classify MRI data 
* derived functional connectome (fMRI series + T1-weighted MRI)