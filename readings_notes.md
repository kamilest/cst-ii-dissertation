# Readings notes

## A parameter-efficient deep learning approach to predict conversion from mild cognitive impairment to Alzheimer's disease

* identifying MCI patients with a high likelihood of developing AD within 3 years
* *multitasking* deep learning algorithm: predicting both MCI conversion and AD vs healthy controls
* fewer parameters limit overfitting to the dataset
  * batch normalisation (whitening)
  * dropout
  * L2 regularisation (penalises high absolute value of weights)
* most predictive parameters: 
  * structural MRI images
  * demographic, neuropsychological data
  * APOe4 (genetics)
* performance (MCI to AD conversion)
  * 0.925 AUC
  * 10cv accuracy of 86%
  * sensitivity 87.5%
  * specificity of 85%
* perfect performance distinguishing controls and affected subjects
* *robustness* to low information images
* *separable convolutions* are a modification of a normal convolution that splits it into two steps, and reducing the number of parameters (decrease overfitting)
* not negatively affected by inclusion of irrelevant features (discussion section uses this in discussing *why* deep learning is better than alternative previously used approaches like SVMs)
* listing why the algorithm was innovative [more important in research than in a Part II dissertation]
* use case: helping to diagnose patients—not as feasible with the static GCN because the model must be retrained when every patient is added.

What?
> Around 10%–15% of MCI patients per year convert to AD over a relatively short time (Braak and Braak, 1995; Mitchell and Shiri-Feshki, 2008), although the annual conversion rate tends to progressively diminish. The mean conversion rate from MCI to AD is approximately 4% per year.

### Data preprocessing
* common space (use templates for MRI imaging)
  * two templates for measuring robustness
  * >symmetrical diffeomorphic mapping and employed five total iterations
    * center and mass alignment, rigid, similarity and fully affine transformations; sampling, neighbourhood cross-correlation
    * winsorisation, numerical normalisation
    * smoothing signals, shrink factors, resolution levels
  * MNI152_T1_1mm template
* local Jacobian Determinant images of nonlinear part of deformational field [???]
* definition of regions of interest (ROIs)
* ANTs package (Avants et al., 2010, 2011)


**Evaluation of the classifier:**
* sampling strategy: divide samples in training/validation/test set splits
  * talking about the percentage splits (stratified(?) by subjects), whether data augmentation was used,...
* accuracy ACC, sensitivity SEN, specificity SPE, receiving operating curve ROC AUC
* Mann-Whitney U test
* robustness of network to structural misalignment in the MNI space
  * *random permutation of training labels*


## Handedness, language areas and neuropsychiatric diseases: insights from brain imaging and genetics

### Data preprocessing
* volumetric, area, thickness measures
* quantile normalisation, temporal synchronisation


## Co-Attentive Cross-Modal Deep Learning for Complex Disease Analysis

* multi-head co-attention (MHCA) CNN, taking in both SPECT image (3D image of brain to model dopamine transporter functionality) and DNA methylation (DNAm) data in separate channels and then combining the extracted features
* go through separate encoders with additional feature selection algorithm on methylation data—XGBoost for getting feature importance weights, recursive feature elimination (RFE) for removing based on those importance weights
* feature importance/attention weights tell which features are considered more important (for *interpretability*)
* train (80%) and hold-out test (20%) sets, 5-fold cross-validation on test set to measure mean and standard deviation of overall evaluation metrics—accuracy, AUC (and additional bonus for small number of parameters)


## Personalised intrinsic network topography mapping and functional connectivity deficits in Autism Spectrum Disorder

* individually specific variation in brain architecture
* >  there is some literature on the notion that resting-state network in autism are topologically slightly different [paper] which might mean that imposing a normal generic template to a disease population (by averaging across nodes within a parcel) may obscure subtle differences. Thus it would actually be interesting to see if using a latent time-series from a parcel instead of a mean also improves disease classification


## Heterogeneous graph attention network
* containing different types of nodes and links
* attention mechanism has a lot of potential and has attracted a lot of attention recently
* hierarchical node- and semantic-based attention mechanism
  * *node-level attention*: learn the importance between node and meta-path neighbours
  * *semantic-level attention*: learn importance between various meta-paths
* generating node embedding by aggregating features from meta-path based neighbours
* previous approaches: *deep neural networks* generating node representation based on features and neighbours; *graph convolutional networks* extend this by applying convolution operator; *attention mechanisms* encouraging to focus on the most informative parts of the structure
* *Graph Attention Network (GAT)* has convolution and attention but only on one type of node or link
* *Heterogeneous Information Network (HIN)*: multiple types of nodes and edges
* *meta-path*: composite relation between two subjects (e.g. co-actor represented by film-actor-film)
  * resembles database design