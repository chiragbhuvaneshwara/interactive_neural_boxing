Utility functions and classes for loading motions from BVH files and editing of motions using a quaternion representation.

This module is supposed to be loaded as a submodule in an experiment or tool environment. 
From a parent folder, you can directly import parts of the model.

E.g. 

/my_experiment/

/my_experiment/experiment_file.py

/my_experiment/mosi_utils_anim/... 


--- experiment_file.py ----

from mosi_utils_anim.preprocessing.feature_extractor import FeatureExtractor
...

