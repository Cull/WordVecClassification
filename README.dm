Word vectors classification:

Dependecies:
  tenserflow --- 1.12.0
  keras --- 2.2.4
  scipy
  numpy
  pickle
  sklearn


For training use:
     ./VectorClassification.py --train_data path_to_train_data.pkl (path pickle train data file)

    model will be saved to trained/ directory

For validation use:
     ./validation.py --trained_model trained/model_trained.h5 --validation_data path_to_validation_data.pkl (path to pickle validation data file)
