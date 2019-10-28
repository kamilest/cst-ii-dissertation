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


## Unsupervised Scalable Representation Learning for Multivariate Time Series 

Franceschi, Dieuleveut, and Jaggi
-------

* few articles explicitly deal with general-purpose representation learning for time series without assuming what the data type actually represents
* *unsupervised* representation for rarely or sparsely labelled time series
* *compatible representations* for *unequal time lengths* (general-purpose, multivariate)
* *scalability and efficiency*
* scalable encoder—deep CNN with idlated convolutions
* *triplet loss*:
  * time-based negative sampling
  * advantage of encoder resilience to unequal time lengths
* tested on multiple datasets for ensured universality: *generality*, *transferability, outperforming concurrent methods, matching state-of-the-art*


Related work
* unsupervised learning for time series
* triplet losses (for representation learning, but never in a fully unsupervised setting)
* convolutional networks for time series (including dilated convolutions)

Unsupervised training
* encoder-only architecture
* triplet loss *for time series* (inspired by representation learning framework word2vec)
* time-based sampling strategies to overcome the challenge of learning on unlabeled data
* similar time series should obtain similar representations
* negative sampling: inspired by word2vec
  * take a time series pieces of which should be similar to each other and have similar representations (as they have the same context) but comparing to another random time series should be generally not similar (because they probably refer to the different context)
  * taking subseries of a given time series, representation of subseries should be similar to the representations of different subseries of the same time series (*positive* example)
  * the subseries of a given time series should have a different representation from the one of a randomly chosen subseries from any time series (*negative* example)
  * so triplet means: *reference*, *positive example*, *negative example*

$$
-\log\left( \sigma \left( f(x^{\mathrm{ref}}, \theta)^{\mathrm{T}}\ f(x^{\mathrm{pos}}, \theta)\right)\right) - \sum\limits_{k=1}^{K}\log\left( \sigma \left( -f(x^{\mathrm{ref}}, \theta)^{\mathrm{T}}\ f(x^{\mathrm{neg}}_k, \theta)\right)\right)
$$

Encoder architecture motivated by 
* extraction of relevant information
* time/memory efficient
* variable-length inputs
  
Convolutional networks
* seem to be solution for the three requirements above
* good for parallelisation on GPUs (unlike recurrent)
* *exponentially dilated* convolutions work better in capturing *long-range dependencies* 
* *causal* convolutions
  * map a sequence to a sequence of the same length such that $i$th element of the output sequence is computed using only values up to until the $i$th element of the input sequence for all $i$.
  * alleviate disadvantage of not using recurrent networks

Testing
* comparison of classification performance to other state-of-the-art supervised and unsupervised networks in addition to assessing transferability of the representation.
* for each dataset with a *train/test split* (no hyperparameter optimisation), unsupervisedly train an encoder using train set; train a SVM with a radial basis function kernel on top of learned features using the train labels of the dataset
* simple SVM checks if the encodings are separable; when encoder is trained SVM allows efficient training in terms of time and space
  

## Brain aging comprises multiple modes of structural and functional change with distinct genetic and biophysical associations

Smith, Elliott, Alfaro-Almagro et al.

Suggested reading by Richard
------------
### Abstract **quotes**
* Brain imaging for studying how brains are aging compared against population norms
* brain health aspects: some factors can accelerate brain tissue aging
* functional and structural brain change
* association with genetics, lifestyle, cognition, physical measures and disease
* many modes had genetic associations

### Introduction
* brain age: apparent age of individuals' brains
* difference between brain age and chronological age (brain age gap/delta)
* e.g. atrophy in MRI data would suggest brain is older than the normal age-matched brain
* aging and disease: AD has patterns of accelerated healthy aging
* structural preprocessing: *wrapping* to a standard space, *gray matter* segmentation, *voxelwise segmentation*
* *supervised* learning algorithms (regression, SVM, deep learning)
* more modalities other than structural (geom layout of the brain): structural *connectivity*, white matter mictrostructure, *functional* connectivity, iron deposition, cognitive task activation
* > richer range of structural and functional measures of change in the brain
* influenced by multiple biological processes (exercise, diet, smoking, other health factors)—whose influence could also be different depending on age
* > cost of losing important information regarding distinctions between multiple biological factors
* 62 distinct modes of population variation
* 21407 participants over the age of 45
* 4 sites identical imaging hardware, scanner software and protocols
* genetics, lifestyle, cognitive and physical measures, healthcare info
* 3913 IDP (imaging-derived phenotypes)—summary measures describing different aspect of brain structure/function
  * functional/structural connectivity
  * tissue microstructure
  * geometry of cortical/subcortical structures
* identify multiple modes representing different combinations of IDPs—use them separately: large number of distinct brain age predictions
* ICA—decompose IDP data into 62 modes of variation
* subject-weight vector (???)
* association tests against non-imaging variables and genetics
* some modes show genetic association with brain aging, others could represent other factors that influence brain age

### Results
Code: https://www.fmrib.ox.ac.uk/ukbiobank/BrainAgingModes

* discarded outliers/missing data
* retained 18707 subjects
* 62 ICA modes
* modes were inverted sometimes so that correlation with age is positive
* reordered for decreased variance
* similar behaviour for females/males
* mean absolute delta of 2.9 years
* > unique variance is also referred to as the "partialled" mode, which is calculated by takinga mode's subject weight vector and regressing out the subject vectors of all other modes
  * ???
  * examine associations with non-imaging variables as the unique subject variance by a given mode is isolated
  * contribution to age modelling varies highly from mode to mode
* Bonferroni correction for deriving correlation with age ($p < 0.05/62$)
* mode clustering
* brain aging modes mapped back to structure and function (what do those ICA modes represent???)
* 8787 nIDPs; 16 groups of variable types (maternal smoking, birth weight; exercise, diet, alcohol+tobacco; body size, fat, bone density; cognitive tests; health outcome—diagnosis)
* Table 2 displays the meanings of clusters having an impact on ageing. Not very explanatory imo because the factors included in one cluster are not even that related (e.g. thalamus volume and body size); especially when some clusters interpret the *same* variable as having either positive or negative correlation
  * I guess you have to take the *whole cluster* into account as having an effect on accelerated brain aging? e.g. BMI does not mean anything by itself unless you *also* smoke (one effect) or have a small T1w grey-white contrast?
  
### Discussion
* > Kaufmann et al. 2019 used a single imaging modality (T1-weighted structural images) from 45,000 subjects pooled from 40 studies, to investigate the relationship between brain aging and several diseases. Brain-age prediction was trained from whole-brain analysis of the structural data, and also seven atlas-defined regional subsets were used to retrain the predictions. The different regional brain-age delta estimates showed varying associations with disease. However, as with our all-in-one prediction and also Ning et al., 2018, direct GWAS of the delta estimates showed virtually no significant association, even with these high subject numbers.
* > value in considering multiple, multimodal brain aging modes separately
  * e.g. all-in-one estimate had no genetic influence but individual modes had significant association with genetic variables
  * non-imaging and non-genetic: bone density, body size and fat, metabolic/cardiovascular function, blood pressure, haemoglobin, age at menopause; life factors: alcohol, (maternal) smoking, exercise, sleep duration; cognitive test scores (processing speed, IQ); mental health; disease (e.g. diabetes)
* not all diseases display patterns identical to accelerated normal brain aging; could detect disease sub-groups; detect disease effects vs non-disease aging effects
* number of modes have unique variance negatively correlated 
* brain age gap has potential weakness of assuming that the offset would be constant for a given subject as the subject gets older
* but could instead be e.g. "a given subject's brain is aging faster than the population average in terms of fixed distinct aging rate, implying that the delta would be increasing over time"
