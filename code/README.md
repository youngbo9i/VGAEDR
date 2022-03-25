# Predicting novel drug-disease associations for drug repositioning through variational infer-ence and graph autoencoder
Code for our paper“”
## DataSets
* data/drug_dis.csv is the drug_disease association matrix, which contain 18416 associations between 269 drugs and 598 diseases.
* data/drug_sim.csv is the drug similarity matrix of 269 diseases,which is calculated based on drug target features.
* data/dis_sim.csv is the disease similarity matrix of 598 diseases,which is calculated based on disease mesh descriptors.
## Requirements
The code has been tested running under Python 3.8.0, with the following packages and their dependencies installed:
###
    numpy==1.21.5
    torch==1.8.0
## Usage
    python main.py --data 1