# Data Preparation module

In this module, ``start.py`` begins the data preparation process. Use following command from project root directory:

``python -m data.neural_data_prep.start``

This script needs as input:
1. The raw motion capture data found in ``data/raw_data/mocap/hq/processed``.
2. Punch labels in ``data/raw_data/punch_label_gen/punch_label`` generated through blender for the above mentioned mocap data.

This script produces output in ``data/neural_data``. The following is produced as output:
1. A numpy file containing the input and output pairs for the neural network training.
2. A json file containing information about the parameters used to generate the data and the demarcation for the different 
variables in the input and output vectors.