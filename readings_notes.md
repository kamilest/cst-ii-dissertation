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


**Evaluation of the classifier:**
* sampling strategy: divide samples in training/validation/test set splits
  * talking about the percentage splits (stratified(?) by subjects), whether data augmentation was used,...
* accuracy ACC, sensitivity SEN, specificity SPE, receiving operating curve ROC AUC
* Mann-Whitney U test
* robustness of network to structural misalignment in the MNI space
  * *random permutation of training labels*

