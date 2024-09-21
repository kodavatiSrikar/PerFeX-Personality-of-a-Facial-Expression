# PerFeX: Personality of a Facial Expression


## Overview

This work presents a model that identifies a person's personality traits given facial expressions, and a large-scale standardized dataset of facial expressions in terms of Action Units. The project employs an iterative training approach to train two models (a Hybrid Attention model and a 1D-CNN model) that predict personality traits based on the Action Units.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing. The dataset can be generated using two models provided in the project.

### Prerequisites

What things you need to install and how to install them:

- Python 3.8+

### Installation

A step-by-step series of examples that tell you how to get a development environment running:


**Clone the repository:**
   ```bash
   git clone https://github.com/your_username/your_project.git](https://github.com/kodavatiSrikar/Dataset-for-facial-expression-of-personality.git
   cd Dataset-for-facial-expression-of-personality
   ```

Install the requirements using the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install -r requirements.txt
```

## Downloading Files from Google Drive

To download the necessary files from Google Drive, follow these steps:

1. Copy the file's sharing link from Google Drive.
   [Dataset](https://drive.google.com/drive/folders/1n9G8FeW_8PeC4JbNj_1vFRuOEVretTGw?usp=drive_link)
2. Download the folder, unzip the files inside, and copy the files inside the Dataset-for-facial-expression-of-personality folder.

## Usage

Follow the below sections to train and run the inference of the models. 

## [Note]

Inference of the model can be executed without training the model. The pre-trained weights for both models are provided in Google Drive.

## Data Augumentation
Augment the training data
```bash
python data_augmentation.py
```

## Multi-scale CNN

## Training

Run the following in the project directory to train the CNN model

```bash
python 1dcnn_train.py
```

## Retraining

Retraining the CNN model with the data obtained from the user study.

```bash
python 1dcnn_retrain.py
```

## Testing

Run the following in the project directory to test the CNN model performance.

```bash
python 1dcnn_test.py
```

## Deployment

Run the following in the project directory to generate the personality traits data using the CNN model.

```bash
python 1dcnn_deploy.py
```

## Hybrid Model

## Training

Run the following in the project directory to train the hybrid model

```bash
python attn_train.py
```


## Retraining

Retraining the hybrid model with the data obtained from the user study.

```bash
python attn_retrain.py
```
## Testing

Run the following in the project directory to test the hybrid model performance.

```bash
python attn_test.py
```

## Deployment

Run the following in the project directory to generate the personality traits data using a hybrid model.

```bash
python attn_deploy.py
```

## Custom action units

Action Units can be extracted from custom videos using the OpenFace library, which employs the FACS principle. Please use the following [Documention](https://github.com/TadasBaltrusaitis/OpenFace/wiki) to obtain the Action Units used to input our model.


