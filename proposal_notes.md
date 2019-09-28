# Project proposal notes

## Inspiration for the project

The papers and the code (described below) describe a semi-supervised graph convolutional network (GCN) to predict  
* whether the patient suffers from Autism Spectrum Disorder (ASD) and
* whether the patient suffering from Mild Cognitive Impairment (MCI) will develop Alzheimer's disease (AD).

The graph is used to exploit both 
* *the individual features* of patients (stored as vertex features in the graph) and the 
* *similarity between patients* (represented as possibly weighted edges in the graph). 

This is thought to be better than using just graphs (which ignore individual features of patients) or just non-graph classifiers (which would not allow for patients to share the data and infer the diagnosis from patient's neighbourhood, especially when imaging data is not available for all patients). The graph also allows to exploit *multimodality* of data—the approach uses both imaging (fMRI, MRI, brain volume, longitudinal brain scans etc.) and non-imaging (gender, age, acquisition site (determining the imaging data collection protocol), possibly genetic) data.

### Papers
The main papers on which the project would be based are:

Parisot, S., Ktena, S. I., Ferrante, E., Lee, M., Moreno, R. G., Glocker, B., & Rueckert, D. (2017). <br/>
[Spectral Graph Convolutions for Population-based Disease Prediction](https://arxiv.org/abs/1703.03020). <br/>
MICCAI 2017.

and 

*Parisot, S., *Ktena, S. I., Ferrante, E., Lee, M., Moreno, R. G., Glocker, B., & Rueckert, D. (2017). <br/>
[Disease Prediction using Graph Convolutional Networks: Application to Autism Spectrum Disorder and Alzheimer’s Disease](https://arxiv.org/abs/1806.01738). <br/>
Medical Image Analysis, 2018.

### Codebase
The code used for the papers above (applied only to ABIDE dataset for ASD but not ADNI for AD) is publicly available at https://github.com/parisots/population-gcn. The implementation for AD, however, should be similar (and would need coding the data parser and processor from scratch).

This GCN is implemented using TensorFlow which does not support deep learning on graph structured data as well as some other libraries like `pytorch_geometric` (PyG) or Deep Graph Library (DGL) would, and so is less extensible/applicable to other problems and datasets. 

The above implementation is custom-built for those specific papers, which in turn is adapted from https://github.com/tkipf/gcn.


### Datasets
The project will primarily only use ADNI for Alzheimer's disease as the benchmark, but some ideas could incorporate such datasets as ABIDE (ASD) and PPMI (Parkinson's disease) as alternative datasets/benchmarks.

* ADNI (Alzheimer's Disease Neuroimaging Initiative): http://adni.loni.usc.edu [*Access requested and retreived*]
* ABIDE (Autism Brain Imaging Data Exchange): [http://fcon_1000.projects.nitrc.org/indi/abide/](http://fcon_1000.projects.nitrc.org/indi/abide/)
* PPMI (Parkinson's Progression Markers Initiative): https://www.ppmi-info.org/

## Reimplement + extend
This seems to be a common approach to Part II projects and I like that it gives some direction to what should be achievable (i.e. at least reproducing the results). 

Addressing some problems with the current implementation could lead to new state-of-the-art results while making the API more accessible and extensible.

### Using another geometric deep learning library
Original paper implemented in TensorFlow with a custom implementation of a GCN (indicating the lack of common TensorFlow APIs for this kind of task).
* [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/en/latest/)
* [DGL](https://www.dgl.ai/pages/about.html)

## Apply to another problem

## Apply to another dataset

