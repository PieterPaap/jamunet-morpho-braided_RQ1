import os
import torch 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
from osgeo import gdal 
from torch.utils.data import TensorDataset 

from preprocessing.satellite_analysis_pre import count_pixels
from preprocessing.satellite_analysis_pre import load_avg

def load_image_array(path, scaled_classes=True):
    '''Load a single image using Gdal and convert to float32 array.'''
    ds = gdal.Open(path)
    if ds is None: return None
    img_array = ds.ReadAsArray().astype(np.float32)

    if scaled_classes:
        img_array = img_array.astype(int)
        img_array[img_array==0] = -1
        img_array[img_array==1] = 0
        img_array[img_array==2] = 1
    
    return img_array

def load_avg(train_val_test, reach, year, dir_averages):
    '''Robustly loads the average image for a given year.'''
    # 1. Try standard path
    folder_name = f'average_{train_val_test}_r{reach}'
    filename = f'average_{year}_{train_val_test}_r{reach}.csv'
    full_path = os.path.join(dir_averages, folder_name, filename)
    
    if os.path.exists(full_path):
        return pd.read_csv(full_path, header=None).to_numpy()
    
    # 2. Fallback: Try finding it in generic folders
    potential_folders = [f'average_training_r{reach}', f'average_r{reach}']
    for folder in potential_folders:
        path_to_check = os.path.join(dir_averages, folder, filename)
        if os.path.exists(path_to_check):
            return pd.read_csv(path_to_check, header=None).to_numpy()
            
    return None

def create_list_images(train_val_test, reach, dir_folders, collection):
    '''Finds the correct reach folder and returns list of .tif images.'''
    list_dir_images = []
    
    if not os.path.exists(dir_folders):
        print(f"Error: Directory not found: {dir_folders}")
        return []

    target_folder_path = None
    
    # Search for folder ending in "_rX"
    for folder_name in os.listdir(dir_folders):
        if folder_name.endswith(f'_r{reach}'):
            # If a specific filter (like 'training') is requested, prioritize it
            if train_val_test in folder_name or train_val_test == 'training':
                target_folder_path = os.path.join(dir_folders, folder_name)
                break
    
    # Fallback: Just match reach ID if specific tag not found
    if target_folder_path is None:
        for folder_name in os.listdir(dir_folders):
            if folder_name.endswith(f'_r{reach}'):
                target_folder_path = os.path.join(dir_folders, folder_name)
                break

    if target_folder_path is None:
        return []

    # Collect images
    sorted_files = sorted(os.listdir(target_folder_path))
    for image in sorted_files:
        if image.endswith('.tif'):
            list_dir_images.append(os.path.join(target_folder_path, image))
            
    return list_dir_images

def create_datasets(train_val_test, reach, year_target=5, nodata_value=-1, dir_folders=r'data\satellite\dataset', 
                    collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True):
    
    # 1. Get List of Image Paths
    list_dir_images = create_list_images(train_val_test, reach, dir_folders, collection)
    if not list_dir_images: 
        return [], [], [], [] 

    # 2. Load Images into Arrays and Extract Years
    images_array = []
    loaded_years = []
    
    for idx, path in enumerate(list_dir_images):
        img = load_image_array(path, scaled_classes=scaled_classes)
        if img is not None:
            images_array.append(img)
            filename = os.path.basename(path)
            try:
                # Extract 4-digit year from filename (e.g., 1988_03_01...)
                parts = filename.replace('-', '_').split('_')
                year = next(p for p in parts if p.isdigit() and len(p) == 4)
                loaded_years.append(int(year))
            except StopIteration:
                # Fallback if no 4-digit number is found
                loaded_years.append(1988 + idx)

    # 3. Load Averages (Dynamic Path Logic)
    # Navigates: month_X -> preprocessed -> {River}_images -> averages
    parent_dir = os.path.dirname(dir_folders)
    river_base_dir = os.path.dirname(parent_dir)
    dir_averages_dynamic = os.path.join(river_base_dir, 'averages')

    avg_imgs = []
    for year in loaded_years:
        avg = load_avg(train_val_test, reach, year, dir_averages=dir_averages_dynamic)
        if avg is None:
            # If no average found, use a zero-array to allow processing to continue
            if len(images_array) > 0: 
                avg = np.zeros_like(images_array[0])
            else: 
                return [], [], [], []
        avg_imgs.append(avg)

    # 4. Replace No-Data (Infilling)
    # Replaces pixels with value -1 (no-data) with values from the seasonal average
    good_images_array = [np.where(image == nodata_value, avg_imgs[i], image) 
                         for i, image in enumerate(images_array)]
        
    # 5. Create Sequences (n-to-1 sliding window)
    input_dataset = []
    target_dataset = []
    years_out = []
    reaches_out = []
    
    if len(good_images_array) < year_target: 
        return [], [], [], [] 

    for i in range(len(good_images_array) - year_target + 1):
        # Input is a sequence of length year_target - 1
        input_dataset.append(good_images_array[i : i + year_target - 1])
        # Target is the single image following the sequence
        target_dataset.append([good_images_array[i + year_target - 1]])
        
        # Metadata tracking
        years_out.append(loaded_years[i])
        reaches_out.append(reach)

    return input_dataset, target_dataset, years_out, reaches_out

def combine_datasets(train_val_test, reach, year_target=5, nonwater_threshold=480000, nodata_value=-1, nonwater_value=0,   
                     dir_folders=r'data\satellite\dataset', collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True):
    
    # 1. Create the raw sequences and get metadata
    input_dataset, target_dataset, years, reach_ids = create_datasets(
        train_val_test, reach, year_target, nodata_value, 
        dir_folders, collection, scaled_classes
    )

    filtered_input_dataset = []
    filtered_target_dataset = []
    filtered_years = []
    filtered_reaches = []

    # 2. Filtering logic (similar to old version but more robust)
    for input_images, target_image_seq, year, r_id in zip(input_dataset, target_dataset, years, reach_ids):
        is_input_good = True
        for img in input_images:
            # Check if image exceeds non-water threshold
            if np.sum(img == nonwater_value) >= nonwater_threshold:
                is_input_good = False
                break
        
        if is_input_good:
            target_img = target_image_seq[0]
            if np.sum(target_img == nonwater_value) < nonwater_threshold:
                filtered_input_dataset.append(input_images)
                filtered_target_dataset.append(target_img)
                filtered_years.append(year)
                filtered_reaches.append(r_id)

    return filtered_input_dataset, filtered_target_dataset, filtered_years, filtered_reaches

def create_full_dataset(train_val_test, year_target=5, nonwater_threshold=480000, nodata_value=-1, nonwater_value=0, 
                        dir_folders=r'data\satellite\dataset', name_filter=None,
                        collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True, device='cuda:0', 
                        dtype=torch.float32):
    
    all_inputs = []
    all_targets = []
    all_years = []
    all_reaches = []
    
    if not os.path.exists(dir_folders):
        print(f"Path not found: {dir_folders}")
        return None

    potential_folders = [f for f in os.listdir(dir_folders) if os.path.isdir(os.path.join(dir_folders, f))]
    
    for folder_name in potential_folders:
        # Match folder based on usage (train/val/test)
        if name_filter and name_filter not in folder_name:
            continue
        elif not name_filter and train_val_test not in folder_name:
            continue

        try:
            reach_id_int = int(folder_name.split('_r')[-1])
        except:
            continue
            
        use_label = name_filter if name_filter else train_val_test
        
        # Get filtered data for this reach
        inputs, targets, years, reaches = combine_datasets(
            train_val_test=use_label, 
            reach=reach_id_int, 
            year_target=year_target, 
            nonwater_threshold=nonwater_threshold,
            nodata_value=nodata_value, 
            nonwater_value=nonwater_value, 
            dir_folders=dir_folders, 
            collection=collection, 
            scaled_classes=scaled_classes
        )
        
        if len(inputs) > 0:
            all_inputs.extend(inputs)
            all_targets.extend(targets)
            all_years.extend(years)
            all_reaches.extend(reaches)

    if not all_inputs:
        print("No valid sequences found.")
        return None

    # --- Convert to Tensors ---
    input_tensor = torch.tensor(np.array(all_inputs), dtype=dtype)
    target_tensor = torch.tensor(np.array(all_targets), dtype=dtype)
    years_tensor = torch.tensor(all_years, dtype=torch.int64)
    
    # Ensure reaches are processed as integers
    reaches_processed = [int(str(r).replace('r','')) for r in all_reaches]
    reaches_tensor = torch.tensor(reaches_processed, dtype=torch.int64)

    # Return Dataset without CI scores
    return TensorDataset(input_tensor, target_tensor, years_tensor, reaches_tensor)