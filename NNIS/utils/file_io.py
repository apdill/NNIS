# nn_is/utils/file_io.py

import numpy as np
import os
import pickle
import tifffile as tiff

def save_masks(network, output_dir='output'):
    """
    Saves the masks of individual neurons and the combined network mask to TIFF files.

    Args:
        network (Network): The network whose masks are to be saved.
        output_dir (str): Directory to save the mask files.
    """
    os.makedirs(output_dir, exist_ok=True)
    combined_mask = np.zeros((network.height, network.width), dtype=np.uint16)

    for idx, neuron in enumerate(network.neurons, start=1):
        neuron_mask = neuron.generate_binary_mask()
        # Save individual neuron mask as TIFF
        filename = f"{network.network_id}_neuron_{idx}_mask.tiff"
        filepath = os.path.join(output_dir, filename)
        tiff.imwrite(filepath, neuron_mask.astype(np.uint8), compression='deflate')

        # Add to combined mask with unique labels
        combined_mask[neuron_mask > 0] = idx

    # Save the combined mask as TIFF
    combined_mask_filename = f"{network.network_id}_combined_mask.tiff"
    combined_mask_path = os.path.join(output_dir, combined_mask_filename)
    tiff.imwrite(combined_mask_path, combined_mask.astype(np.uint16), compression='deflate')

    # Save the network image as TIFF
    network_image_filename = f"{network.network_id}_image.tiff"
    network_image_path = os.path.join(output_dir, network_image_filename)
    tiff.imwrite(network_image_path, network.network_mask.astype(np.uint8), compression='deflate')

def save_dataframe(network, df, output_dir='output'):
    """
    Saves the pandas DataFrame containing the masks to a pickle file.

    Args:
        network (Network): The network corresponding to the DataFrame.
        df (DataFrame): The DataFrame to be saved.
        output_dir (str): Directory to save the DataFrame file.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataframe_filename = f"{network.network_id}_masks_dataframe.pkl"
    dataframe_path = os.path.join(output_dir, dataframe_filename)

    # Save the DataFrame using pickle
    with open(dataframe_path, 'wb') as f:
        pickle.dump(df, f)
