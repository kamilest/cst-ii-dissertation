# Discussions of the neuroscientific datasets
[Richard]

Just a quick follow-up on the meeting:

- The brain hack site just went live: https://oxbridgebrainhack.github.io
- Re Alex's side project idea there is some literature on the notion that resting-state network in autism are topologically slightly different (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6076333/) which might mean that imposing a normal generic template to a disease population (by averaging across nodes within a parcel) may obscure subtle differences. Thus it would actually be interesting to see if using a latent time-series from a parcel instead of a mean also improves disease classification. Just a thought :)

[Alex]

Thank-you for the link. When Pietro is back I can float the idea to him as a potential project.

Mechanically for this to work, I need to understand a little bit more when you say you register the atlas to the brain in your work:

1) Does this mean you are registering MNI space atlas to native space brain?
2) If 1) is true, then does this mean that the same parcel across multiple individuals might contain a different number of voxels?

If initial results look promising, the convolutions can be replaced with wavelet scatterings, and/or either method augmented with spatial-attention to see which sub time-series in a parcel gives the highest weighting to the single latent series. 

If the results do not look good, then I think it is possible to define a new (functional) atlas in a totally unsupervised way using a triplet loss function paradigm which I saw in [this recent paper](https://arxiv.org/pdf/1901.10738.pdf). With some adaptions that I have briefly worked through, it would be possible to group the voxel time series that are most ’similar’ without any prior supervision to learn such similarity. Spatial priors would help regularise the search to ensure it actually made sense in terms of brain anatomy.

This latter project is larger and more ambitious and (potentially an MPhil project) but it would mean the birth of a new ‘data driven’ atlas. Maybe even better than the Glasser if multi-modal data can be integrated.


[Richard]

1. Yes we warp a template to native space (though when I work in freesurfer we don't exactly used MNI, but fsaverage)
2. Yes that is possible. An example freesurfer output is attached. As you can see there is a column that indicates the number of vertices included within a given parcel/ROI.

If Matt Glasser did his job correct then the parcels they found provide the optimal subdivision based on multimodal data, but this doesn't mean that this parcellation (which is based on a normative largely adolescent sample) provides a one-size fits all solution and I don't think it completely excludes the possibility that some areas within a parcel actually contribute more to the signal then others in which case a latent time-series might provide a better fit.

Agreed that if the results do not look good it could be interesting to define a new atlas (its definitely a way to boost your citation score), so whatever the results look like I think there would be an interesting and publishable outcome.

# Project proposal draft

Regarding your comment in the ‘Work to be done’ section for data preprocessing, it means two very different things for someone working in neuroimaging vs machine learning. 'Preprocessed data' for neuroimaging means that the data will be registered (aligned), denoised (remove motion artefacts), and so on. The data will be in a neuroimaging format such as .nii or .nii.gz for it to be viewed in medical imaging software. This is the most likely format you will receive the data in, but it will still require further preprocessing for it to be useable in a machine learning context such as scaling, min-max normalisation, correlation, partial correlation and so on. It is this latter meaning of ‘preprocessed’ that will be a part of this project, as these choices might have a large effect on your experimental outcome! In summary, from a machine learning perspective, there will always be some sort of preprocessing left to do.

