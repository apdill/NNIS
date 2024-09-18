# nn_is/utils/file_io.py

import numpy as np
import os
import pickle
import tifffile as tiff

def save_masks(network_ds, images_dir, masks_dir):
    """
    Saves the masks of individual neurons and the combined network mask to TIFF files.

    Args:
        network_ds (xr.Dataset): The xarray Dataset containing the network data.
        images_dir (str): Directory to save the network images.
        masks_dir (str): Directory to save the mask files.
    """

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Extract the network ID from the dataset attributes
    network_id = network_ds.attrs['network_id']

    # Save the network mask
    network_mask = network_ds[f'{network_id}_network_mask'].values
    network_image_filename = f"{network_id}_image.tiff"
    network_image_path = os.path.join(images_dir, network_image_filename)
    tiff.imwrite(network_image_path, (network_mask * 255).astype(np.uint8), compression='deflate')

    # Initialize the combined mask
    combined_mask = np.zeros_like(network_mask, dtype=np.uint16)

    # Save individual neuron masks and update combined mask
    neuron_indices = {}
    neuron_idx = 1  # Start labeling neurons from 1
    for var_name in network_ds.data_vars:
        if 'neuron' in var_name:
            neuron_mask = network_ds[var_name].values
            # Save individual neuron mask
            neuron_mask_filename = f"{var_name}_mask.tiff"
            neuron_mask_path = os.path.join(masks_dir, neuron_mask_filename)

            # Ensure neuron mask is binary and scaled properly
            tiff.imwrite(neuron_mask_path, (neuron_mask * 255).astype(np.uint8), compression='deflate')

            # Update combined mask
            combined_mask[neuron_mask > 0] = neuron_idx
            neuron_indices[var_name] = neuron_idx
            neuron_idx += 1

    # Save the combined mask
    combined_mask_filename = f"{network_id}_combined_mask.tiff"
    combined_mask_path = os.path.join(masks_dir, combined_mask_filename)
    tiff.imwrite(combined_mask_path, combined_mask.astype(np.uint16), compression='deflate')

    # Optionally, save a mapping of neuron IDs to indices
    mapping_filename = f"{network_id}_neuron_mapping.txt"
    mapping_path = os.path.join(masks_dir, mapping_filename)
    with open(mapping_path, 'w') as f:
        for neuron_id, idx in neuron_indices.items():
            f.write(f"{neuron_id}: {idx}\n")

def save_dataset(ds, dataframes_dir):
    """
    Saves the xarray Dataset containing the masks and network parameters to a NetCDF file.
    
    Args:
        ds (xr.Dataset): The Dataset to be saved.
        dataframes_dir (str): Directory to save the Dataset file.
    """
    
    os.makedirs(dataframes_dir, exist_ok=True)
    network_id = ds.attrs['network_id']
    dataset_filename = f"{network_id}_dataset.nc"
    dataset_path = os.path.join(dataframes_dir, dataset_filename)
    
    # Save the Dataset using NetCDF format
    ds.to_netcdf(dataset_path)
