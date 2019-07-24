# MOSI Dev VINN

This repository contains code for the training, evaluation and execution of PFNN - like neural networks. 

Please consider using the Git Workflow: 

* Master-branch: contains the latest stable version of this repository
* Develop-branch: contains features staged for the next stable version
* Feature-branches: development branches for individual features, this contains unsafe code


$ conda activate base ; conda create -n "tmp_environment" -y python=3.7 --file .\requirements.txt ; conda activate tmp_environment
 
Executing PFNN server:

$ python .\nn_console.py -x pfnn_np -d .\data\dataset_4DControll.json -o .\trained_models\4DControl\epoch_49.json
