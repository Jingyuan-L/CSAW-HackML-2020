# CSAW-HackML-2020

```bash
├── data 
    └── clean_validation_data.h5 // this is clean data used to evaluate the BadNet and design the backdoor defense
    └── clean_test_data.h5
    └── sunglasses_poisoned_data.h5
├── models
    └── anonymous_bd_net.h5
    └── anonymous_bd_weights.h5
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
    └── multi_trigger_multi_target_bd_net.h5
    └── multi_trigger_multi_target_bd_weights.h5
├── architecture.py
└── eval.py // [ !!!!!This eval file has been modified!!!!! Only added 'import tensorflow as tf' ]
```

## I. Dependencies
   1. Python 3.6.9
   2. Keras 2.3.1
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. TensorFlow-gpu 1.15.2
   
## II. Validation Data
   1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing) and store them under `data/` directory.
   2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals each containing 9 images in the validation dataset.
   3. sunglasses_poisoned_data.h5 contains test images with sunglasses trigger that activates the backdoor for sunglasses_bd_net.h5.

## III. About this Project
   1. This is the submission of CSAW-HackML-2020(https://github.com/csaw-hackml/CSAW-HackML-2020) competition.
   2. The detailed information about the code and how this project works is in the `project_report_jl10915.pdf` file.

## IV. How to Run My `badNet_repair.py` File
   1. Running the following command:
   `python badNet_repair.py <test data directory> <bad model directory>`
    E.g., `python badNet_repair.py data/clean_test_data.h5 models/anonymous_bd_net.h5`
   2. The `badNet_repair.py` file will print all the information you need including the classification accuracy of the test data.
   3. If you want test the repaired good net separately, the good net will been saved in 'models/goodNet.h5', you can run:
   `python eval.py <test data directory> <good net model directory>`
    E.g., `python eval.py data/clean_test_data.h5 models/goodNet.h5`
    
   NOTICE: The eval file has been modified!!!!! Only added 'import tensorflow as tf' at the front. Please do not use the original eval.py file.