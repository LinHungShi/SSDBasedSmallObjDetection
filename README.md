# Project Title

Small Obj Detection

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

This project runs on python 2x. The following libraries are required

```
1. numpy
2. pandas
3. matplotlib
4. tensorflow
5. keras
6. opencv
```

## Running the tests

This section will show you how to run the model and tests

### train the model with the dataset

```
python run_context_ssd 2
```

The trained weight will be in weights directory


### validate the model with validation data

```
python run_context_ssd 1 <weight path>
```
by default, the data used for validation are validation data.
If you want to get the predicted values for training data, replace val_labels with train_labels at the bottom of run_context_ssd.py

Two output will be genearted after validation
1. a csv file in the images_subset directory containing predicted boudning box location and the ground truth
2. images in drawed_images containing validation images

### shortcut

to train the model, run "bash train.sh"

to validate the model, run "bash val.sh"

to create the model, call buildModel function in run_context_ssd.py

to predict the boudning box, call predict function in run_context_ssd.py
