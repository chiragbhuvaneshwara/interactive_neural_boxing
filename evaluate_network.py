
"""
Basically load the data and run it through the network.


"""

import numpy as np


def _store_predictions(network_predicted_output, actual_output):
    output_file_location = "network_output/pfnn_vector_random_noise_each_phase.npz"
    np.savez(output_file_location, network_output=network_predicted_output, actual_output=actual_output)


def evaluate_network(network, dict_data_fields):
    """

    """
    # Load test data
    test_filename = "data/data_mk4D_test.npz"

    test_data_full = np.load(test_filename)

    all_input_motion = test_data_full['Xun']
    all_phase_data = test_data_full['Pun']
    all_output_motion = test_data_full['Yun']

    network_predicted_output_motion = np.zeros_like(all_output_motion)

    num_test_pts = all_input_motion.shape[0]

    for data_idx in range(num_test_pts):
        curr_motion = all_input_motion[data_idx, :]
        curr_phase = all_phase_data[data_idx]
        network_input = [curr_motion, curr_phase]
        network_output = network.forward_pass(network_input)
        network_predicted_output_motion[data_idx, :] = network_output[0]

    _store_predictions(network_predicted_output_motion, all_output_motion)



#############################
# Methods below not used
#############################


def _get_data_params(data_filename):
    full_data = np.load(data_filename)

    output_data_Y = full_data['Yun']

    mean_output_data = np.mean(output_data_Y, axis=0)
    var_output_data = np.var(output_data_Y, axis=0)

    return (mean_output_data, var_output_data)


def evaluate_network_with_random_layers(network_filename, dict_data_fields):
    """

    """
    train_filename = "data/data_mk4D.npz"
    mean_train_data, var_train_data = _get_data_params(train_filename)

    # Initialize the network
    vinn_random = VINN_custom.load(network_filename, mean_train_data, var_train_data, [0])

    # Load test data
    test_filename = "data/data_mk4D_test.npz"

    test_data_full = np.load(test_filename)

    all_input_motion = test_data_full['Xun']
    all_phase_data = test_data_full['Pun']
    all_output_motion = test_data_full['Yun']

    network_predicted_output_motion = np.zeros_like(all_output_motion)

    num_test_pts = all_input_motion.shape[0]

    for data_idx in range(num_test_pts):
        curr_motion = all_input_motion[data_idx, :]
        curr_phase = all_phase_data[data_idx]
        network_input = [curr_motion, curr_phase]
        network_output = network.forward_pass(network_input)
        network_predicted_output_motion[data_idx, :] = network_output[0]

    _store_predictions(network_predicted_output_motion, all_output_motion)




def evaluate_with_random_noise(network, dict_data_fields):
    """

    """
    train_filename = "data/data_mk4D.npz"
    mean_train_data, var_train_data = _get_data_params(train_filename)

    test_filename = "data/data_mk4D_test.npz"

    test_data_full = np.load(test_filename)

    all_input_motion = test_data_full['Xun']
    all_phase_data = test_data_full['Pun']
    all_output_motion = test_data_full['Yun']
    random_predicted_output = np.zeros_like(all_output_motion)

    num_test_pts = all_input_motion.shape[0]
    
    size_output = all_output_motion.shape[1]

    for data_idx in range(num_test_pts):
        epsilon = np.random.normal(0.0, 1.0)
        mean_random = np.random.normal(mean_train_data, var_train_data, size_output)
        var_random = np.random.normal(mean_train_data, var_train_data, size_output)
        random_pred_motion = mean_random + var_random * epsilon
        random_predicted_output[data_idx, :] = random_pred_motion

    _store_predictions(random_predicted_output, all_output_motion)
