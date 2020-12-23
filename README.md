# CSAW-HackML-2020

```bash
├── data 
    └── clean_validation_data.h5 // this is clean data used to evaluate the BadNet and design the backdoor defense
    └── clean_test_data.h5
    └── sunglasses_poisoned_data.h5
    └── anonymous_1_poisoned_data.h5
    └── Multi-trigger Multi-target
        └── eyebrows_poisoned_data.h5
        └── lipstick_poisoned_data.h5
        └── sunglasses_poisoned_data.h5
├── models
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
    └── multi_trigger_multi_target_bd_net.h5
    └── multi_trigger_multi_target_bd_weights.h5
    └── anonymous_1_bd_net.h5
    └── anonymous_1_bd_weights.h5
    └── anonymous_2_bd_net.h5
    └── anonymous_2_bd_weights.h
    └── goodNet_anonymous_1.h5
    └── goodNet_anonymous_1_weights.h5
    └── goodNet_anonymous_2.h5
    └── goodNet_anonymous_2_weights.h
    └── goodNet_sunglasses.h5
    └── goodNet_sunglasses_weights.h5
    └── goodNet_multi_trigger_multi_target.h5
    └── goodNet_multi_trigger_multi_target_weights.h5
├── architecture.py
├── badNet_repair.py // This is the file repairing the bad net!
├── ML_for_Cyber_Security_Project_Report_jl10915.pdf  // This is my project report!
├── eval_sunglass.py // This file is used for evaluating the Submissions
├── eval_anonymous_1.py // This file is used for evaluating the Submissions
├── eval_anonymous_2.py // This file is used for evaluating the Submissions
├── eval_multi_trigger_multi_target.py // This file is used for evaluating the Submissions
└── eval.py //  !!!!!This eval file has been modified!!!!! Only added 'import tensorflow as tf' 

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
   1. This is the submission of [CSAW-HackML-2020](https://github.com/csaw-hackml/CSAW-HackML-2020) competition.
   2. The detailed information about the code and how this project works is in the `ML_for_Cyber_Security_Project_Report_jl10915.pdf` file.

## IV. How to Run My `badNet_repair.py` File
   1. Run the following command:
   
   `python badNet_repair.py <test data directory> <bad model directory>`
   
    E.g., `python badNet_repair.py data/clean_test_data.h5 models/anonymous_bd_net.h5`
    
   2. The `badNet_repair.py` file will print out all the information you need including the classification accuracy of the test data.
   3. If you want test the repaired good net separately, the good net will been saved in 'models/goodNet.h5', you can run:
   
   `python eval.py <test data directory> <good net model directory>`
   
   E.g., `python eval.py data/clean_test_data.h5 models/goodNet.h5`
    
   This eval.py file is design for `.h5` file for calculating the accuracy rate!
    
## V. How to Run My `eval_XXX.py` File
   1. Run the following command(XXX should be replaced by a specific eval file name:
   
   `python eval_XXX.py <test data directory>` 
   
   E.g., `python eval_sunglass.py data/test_image.png`
   
   2. All eval files designed for four net are listed as following:
   
   `eval_sunglass.py`, `eval_anonymous_1.py`, `eval_anonymous_2.py`, `eval_multi_trigger_multi_target.py`.
   
   3. Each scripts automatically load the corresponding repaired good net to classify input image.
   4. Output(will be directly printed out) is either 1283 (if test_image.png is poisoned) or one class in range [0, 1282] (if test_image.png is not poisoned) 
   
   
   
   
   