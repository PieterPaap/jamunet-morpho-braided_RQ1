# This module contains the functions used for plotting and getting information
# on the satellite images available for the chosen datasets

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.dates import DateFormatter
from datetime import datetime 

from preprocessing.satellite_analysis_pre import * 
from preprocessing.images_analysis import show_image_array

# suppress the 'SettingWithCopyWarning' from pandas
pd.options.mode.chained_assignment = None

def get_info_frequency(collection, cloud_cov=100, range_std=3.5, old=False):
         '''
         Function used for calculating the time interval between two consecutive images and getting information on the overall distribution.

         Inputs: 
                collection = str, specifies collection name accordingly to Google Earth Engine nomenclature
                cloud_cov = int, specifies max percentage of cloud cover above which images are discarded. 
                            default: 100, if set to 'None' cloud coverage is undefined and all images are considered
                range_std = int/float, specifies the range of standard deviation above which the value of delta t 
                            is discarded (presence of a big gap)
                            default: 3.5
                old = boolean, specifies whether old charts are loaded - a different name was given to the files
                      default: False, only loads newest data. 

         Outputs: 
                 info = dict, contains four elements (avg, std, min and max)
                 print statements for initial and final statistics and in presence of big gaps (larger than 'range_std' times the std)
         '''
         df = load_df(collection, cloud_cov, print_total=False, old=old)

         filtered_values = df[df['Delta t'] != 0]['Delta t']
         
         days_more = 0
         for i in range(1, len(df)):
              if df['Delta t'].iloc[i] == 0:
                   days_more += 1

         # print initial statistics
         print(f'Max cloud coverage: {cloud_cov}%')
         print(f"Before removal of outliers:\n\
avg: {np.mean(filtered_values).round(2)}\n\
std: {np.std(filtered_values).round(2)}\n\
Days with more than one image: {days_more:.0f}\n")
         
         count_gaps = 0
         for i in range(1, len(df)):
            # remove from statistics observations if larger than 'range_std' times std
            if df['Delta t'].iloc[i] > range_std * np.std(df['Delta t']): 
                count_gaps += 1
                print(f"Gap {count_gaps:.0f}: {df['Delta t'].iloc[i]:.0f} days between {df['Date (yyyy-mm-dd)'].iloc[i-1]} and {df['Date (yyyy-mm-dd)'].iloc[i]}")
                df['Delta t'].iloc[i] = np.nan
         
         filtered_values2 = df[df['Delta t'] != 0]['Delta t']

         # print final statistics
         print(f"\n\
After removal of outliers:\n\
avg: {np.mean(filtered_values2).round(2)}\n\
std: {np.std(filtered_values2).round(2)}\n")
         
         info = {'Average': np.mean(df['Delta t']).round(2), 'Standard deviation': np.std(df['Delta t']).round(2), 
                 'Minimum': np.min(df['Delta t']), 'Maximum': np.max(df['Delta t'])}
         
         return info

def plot_series(collection, cloud_cov=100, time_int=365, old=False):
    '''
    Function used to plot the chart containing the number of satellite images in the specified collection.
    Returns two plots: the upper one contains the number of images available for each day, 
    the lower one shows the cumulative count of images throughout years.

    Inputs:
           collection = str, specifies collection name accordingly to Google Earth Engine nomenclature
           cloud_cov = int, specifies max percentage of cloud cover above which images are discarded. 
                            default: 100, if set to 'None' cloud coverage is undefined and all images are considered
           time_int = int, interval between two consecutive x-ticks (days)
                      default: 365 days = 1 year
           old = boolean, specifies whether old charts are loaded - a different name was given to the files
                 default: False, only loads newest data. 

    Outputs: None - two plots
           
    '''
    df = load_df(collection, cloud_cov=100, print_total=False, old=old)

    # create time array from day of first image to day of last image 
    start_date = datetime(df['Date (yyyy-mm-dd)'].iloc[0].year, df['Date (yyyy-mm-dd)'].iloc[0].month, df['Date (yyyy-mm-dd)'].iloc[0].day)
    end_date = datetime(df['Date (yyyy-mm-dd)'].iloc[-1].year, df['Date (yyyy-mm-dd)'].iloc[-1].month, df['Date (yyyy-mm-dd)'].iloc[-1].day) 
    time_interval = pd.date_range(start=start_date, end=end_date, freq='D') 

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,7))

    fig.suptitle(f"{collection} images - between {df['Date (yyyy-mm-dd)'].iloc[0]} and {df['Date (yyyy-mm-dd)'].iloc[-1]}\n\
Max cloud coverage: {cloud_cov}%")

    ax1.scatter(df['Date (yyyy-mm-dd)'], df['Count'], label='images', color='blue')
    ax1.set_title(f"Single image count per day")
    ax1.set_ylabel('Number of images [-]')
    ax2.set_xlabel('Date (month yyyy)')

    ax2.scatter(df['Date (yyyy-mm-dd)'], df['Cumulative count'], label='images', color='blue')
    ax2.set_title(f"Cumulative image counts - total images: {df['Cumulative count'].iloc[-1]:.0f}")
    ax2.set_ylabel('Cumulative number of images [-]')
    ax2.set_ylim([0, df['Cumulative count'].max()+50])

    date_formatter = DateFormatter('%b %Y')  
    fig.gca().xaxis.set_major_formatter(date_formatter)
    plt.xlim(df['Date (yyyy-mm-dd)'].iloc[0], df['Date (yyyy-mm-dd)'].iloc[-1])
    plt.xticks(ticks= time_interval[0::time_int], rotation=75)
    ax1.legend(), ax2.legend()
    plt.show()

    return None

def show_single_image(collection, train_val_test, year, month, day = 1, reach = 1, img_res=30, 
                      path=r'data\satellite\original', show=True, grayscale=False, vmin=0, vmax=2):
    '''
    Function used for plotting the satellite image given the collection and date.

    Inputs: 
           collection = str, specifies dataset.
                        Available options: 'JRC/GSW1_4/MonthlyHistory', 
                                           'LANDSAT/LT05/C02/T1_L2', 
                                           'COPERNICUS/S1_GRD', 
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           year = int, year of image. Possible range: 1984-2024
           month = int, month of image. Possible range: 1-12
           day = int, day of image.
                 default: 1, otherwise possible range: 1-31
           reach = int, representing reach number. Number increases going upstream.
                   default: 1, applies for both validation and testing.
                   For training, the available range is 1-28 (included)
           img_res = int, image resolution (m).
                     default: 30 m.
           path = str, specifies directory where images are stored
                  default: r'data\satellite\original'
           show = bool, specifies whether the image is shown or not.
                  default: True, set to False in case the image should not be displayed.
           grasyscale = bool, specifies if image is displayed in grayscale.
                        default: False, image displayed in RGB.
           vmin, vmax = int, representing min and max values for image visualization. These only need to be set if image is displayed in grayscale.
                        default: vmin = 0, vmax = 1, applies for JRC collection which has three classes (0: no-data, 1: non-water, 2: water).
                  
    Outputs:
            image = 3D numpy array,
                    dim 0 = length (rows)
                    dim 1 = width (cols)
                    dim 2 = channels (RGB/grey scale)
            plot of the satellite image  
    '''  
    img_path = get_path_images(path, collection, train_val_test, year, month, day, reach)
    image = mpimg.imread(img_path)

    if show==True:
        image_plot = plt.imshow(image) 
        if grayscale==True:
            image_plot_gray = plt.imshow(image, cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax)
        
        plt.title(f'{collection}\n{year}/{month}/{day}')
    
        shp = image.shape
        x_ticks = np.arange(0, shp[1], 300)
        y_ticks = np.arange(0, shp[0], 300)  

        # convert x_ticks and y_ticks from pixels to meters
        x_tick_labels = [round(tick * img_res/1000, 2) for tick in x_ticks]  
        y_tick_labels = [round(tick * img_res/1000, 2) for tick in y_ticks]
        
        plt.xticks(x_ticks, x_tick_labels)
        plt.yticks(y_ticks, y_tick_labels)
        # plt.gca().invert_yaxis() # can be confusing as flow is south-ward directed
        plt.xlabel('Width (km)')
        plt.ylabel('Lentgh (km)')
        plt.show()
    return image

def get_image_shape(collection, train_val_test, year, month, day = 1, reach = 1, img_res=30,
                    path=r'data\satellite\original', show=False, grayscale=False, return_array=False):
    '''
    Function used to get information on the shape of the images. Returns number of rows and columsn, respectively length and width of the image.
    The thirds dimension exists only for RGB images and represents the number of channels of the image.

    Inputs: 
           collection = str, specifies dataset.
                        Available options: 'JRC/GSW1_4/MonthlyHistory', 
                                           'LANDSAT/LT05/C02/T1_L2', 
                                           'COPERNICUS/S1_GRD', 
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           year = int, year of image. Possible range: 1984-2024
           month = int, month of image. Possible range: 1-12
           day = int, day of image.
                 default = 1, otherwise possible range: 1-31
           reach = int, representing reach number. Number increases going upstream.
                   default: 1, applies for both validation and testing.
                   For training, the available range is 1-28 (included)
           img_res = int, image resolution (m).
                     default: 30 m.
           path = str, specifies directory where images are stored
                  default: r'data\satellite\original'
           show = bool, specifies whether the image is shown or not.
                  default: False, set to True in case the image should be displayed (not recommended).
           grayscale = bool, specifies whether image is in RGB or grayscale.
                       default: False (image is RGB), otherwise True (grayscale). 
                       grayscale images have 2D only 
           return_array = bool, specifies whether the function returns the `shape` array.
                          default: False, array is not returned. Set True to return it
    
    Outputs:
            shape = 2D/3D numpy array depending on whether images are in grayscale or RGb, respectively.
                    contains shape of the input image
                    dim 0 = length (rows)
                    dim 1 = width (cols)
                    dim 2 = channels (only for RGB images)
            print statement with shape information.
    '''
    image = show_single_image(collection, train_val_test, year, month, day, 
                              reach, img_res, path, show, grayscale)
    shape = np.shape(image)

    if grayscale == False:
        print(f'RGB image\n\
Collection: {collection}; date: {year}-{month}-{day}\n\
Image rows: {shape[0]}, cols: {shape[1]}, channels: {shape[2]}\n\
Which corresponds to length: {shape[0]*img_res/1000} km and width: {shape[1]*img_res/1000} km')

    if grayscale == True:
        print(f'Grayscale image\n\
Collection: {collection}; date: {year}-{month}-{day}\n\
Image rows: {shape[0]}, cols: {shape[1]}, channel: 1\n\
Which corresponds to length: {shape[0]*img_res/1000} km and width: {shape[1]*img_res/1000} km')
        
    return shape if return_array == True else None

def show_yearly_images(collection, train_val_test, year, day = 1, 
                       reach = 1, img_res=30,  path=r'data\satellite\original'):
    '''
    Function used to plot images of a whole year - can be used with JRC dataset (1 image/month with date always set on 1st day of the month).
    If collection is not JRC_GSW1_4_MonthlyHistory a ValueError is raised. A future improvement will make sure that the function works for all other datasets.

    Inputs: 
           collection = str, specifies dataset.
                        Available options: 'JRC/GSW1_4/MonthlyHistory', 
                                           'LANDSAT/LT05/C02/T1_L2', 
                                           'COPERNICUS/S1_GRD', 
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           year = int, year of image. Possible range: 1984-2024
           month = int, month of image. Possible range: 1-12
           day = int, day of image.
                 default: 1, otherwise possible range: 1-31
           reach = int, representing reach number. Number increases going upstream.
                   default: 1, applies for both validation and testing.
                   For training, the available range is 1-28 (included)
           img_res = int, image resolution (m).
                     default: 30 m.
           path = str, specifies directory where images are stored
                  default: r'data\satellite\original'
           show = bool, specifies whether the image is shown or not.
                  default: True, set to False in case the image should not be displayed.
    
    Outputs:
            none, plot containing the yearly evolution of the images  
    '''
    if collection != r'JRC_GSW1_4_MonthlyHistory':
        raise ValueError(fr'The collection set is {collection}, but this function only works for r"JRC_GSW1_4_MonthlyHistory"') 

    fig, ax = plt.subplots(2, 6, figsize=(20,15),
                           gridspec_kw={'hspace': 0.1, 'wspace': 0.05})

    for i in range(2):
        for j in range(6):
            month = i*6 + j + 1
            
            img_path = get_path_images(path, collection, train_val_test, year, month, day, reach)
    
            image = mpimg.imread(img_path)
            shp = image.shape
            
            x_ticks = np.arange(0, shp[1], 300)
            y_ticks = np.arange(0, shp[0], 300)  

            # convert x_ticks and y_ticks from pixels to meters
            x_tick_labels = [round(tick * img_res/1000, 2) for tick in x_ticks]  
            y_tick_labels = [round(tick * img_res/1000, 2) for tick in y_ticks]

            ax[i,j].imshow(image)
            
            ax[i,j].set_title(f'{month:02d}/{day:02d}', fontsize=13.5) # make sure to print the date with mm/dd format
            # ax[i,j].invert_yaxis() # can be confusing as flow is south-ward directed
            ax[i,j].set_xlabel('Width (km)', fontsize=11)
            ax[i,j].set_xticks(x_ticks)
            ax[i,j].set_xticklabels(x_tick_labels)

            if j == 0:
                ax[i,j].set_yticks(y_ticks)
                ax[i,j].set_yticklabels(y_tick_labels)
                ax[i,j].set_ylabel('Lentgh (km)', fontsize=11) 
            else:
                ax[i,j].set_yticks(y_ticks)
                ax[i,j].set_yticklabels([])         

    fig.suptitle(f'Collection: {collection} - year {year}\nReach {reach} ({train_val_test})', fontsize=16)
    fig.subplots_adjust(top=0.95)  
    plt.tight_layout()
    plt.show()

    return None

def years_evolution(collection, train_val_test, year_start, years_sequence, month, day = 1, 
                    reach = 1, img_res=30, path=r'data\satellite\original', show=True):
    '''
    Function used to the evolution across years in a given month - can be used with JRC dataset (1 image/month with date always set on 1st day of the month).
    If collection is not JRC_GSW1_4_MonthlyHistory a ValueError is raised. A future improvement will make sure that the function works for all other datasets.

    Inputs: 
           collection = str, specifies dataset.
                        Available options: 'JRC/GSW1_4/MonthlyHistory', 
                                           'LANDSAT/LT05/C02/T1_L2', 
                                           'COPERNICUS/S1_GRD', 
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           year_start = int, first year of plotted series. Possible range: 1984-2024
           years_sequence = int, number of years plotted. 
           month = int, month of image. Possible range: 1-12
           day = int, day of image.
                 default: 1, otherwise possible range: 1-31
           reach = int, representing reach number. Number increases going upstream.
                   default: 1, applies for both validation and testing.
                   For training, the available range is 1-28 (included)
           img_res = int, image resolution (m).
                     default: 30 m.
           path = str, specifies directory where images are stored
                  default: r'data\satellite\original'
           show = bool, specifies whether the image is shown or not.
                  default: True, set to False in case the image should not be displayed.
    
    Outputs:
            none, plot containing the yearly evolution of the images  
    '''
    if collection != r'JRC_GSW1_4_MonthlyHistory':
        raise ValueError(fr'The collection set is {collection}, but this function only works for r"JRC_GSW1_4_MonthlyHistory"') 
    
    # if years_sequence > 3:
    #     rows = int(years_sequence)
    
    fig, ax = plt.subplots(1, years_sequence, figsize=(15,30))
    
    for j in range(years_sequence):
        year = year_start + j          
        # if type(month) == int and month < 10:
        #     month = f'0{month}'

        # if type(day)==int and day < 10:
        #     day = f'0{day}'
        
        img_path = get_path_images(path, collection, train_val_test, year, month, day, reach)
    
        image = mpimg.imread(img_path)
        shp = image.shape
        x_ticks = np.arange(0, shp[1], 300)
        y_ticks = np.arange(0, shp[0], 300)  

        # convert x_ticks and y_ticks from pixels to meters
        x_tick_labels = [round(tick * img_res/1000, 2) for tick in x_ticks]  
        y_tick_labels = [round(tick * img_res/1000, 2) for tick in y_ticks]
        
        if show==True:
            ax[j].imshow(image)
            ax[j].set_title(f'{year}/{month:02d}/{day:02d}', fontsize=13.5) # make sure to print the date with yyyy/mm/dd format
            ax[j].set_xticks(x_ticks, x_tick_labels)
            ax[j].set_xlabel('Width (km)', fontsize=11)
            # ax[i,j].invert_yaxis() # can be confusing as flow is south-ward directed
            if j == 0:
                ax[j].set_yticks(y_ticks)
                ax[j].set_yticklabels(y_tick_labels)
                ax[j].set_ylabel('Lentgh (km)', fontsize=11) 
            else:
                ax[j].set_yticks([])         

    # fig.suptitle(f'Collection: {collection}', fontsize=16)
    # fig.subplots_adjust(top=1.45)  
    plt.tight_layout()
    plt.show()

    return None

def plot_image_series(collection, train_val_test, year, month, day=1, reach = 1,
                      img_res=30, path=r'data\satellite\original', num_images=5):
    '''
    Function used to visualize a series of images given the collection and starting date.

    Inputs: 
           collection = str, specifies dataset.
                        Available options: 'JRC/GSW1_4/MonthlyHistory', 
                                           'LANDSAT/LT05/C02/T1_L2',  
                                           'COPERNICUS/S1_GRD', 
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           year = int, year of image. Possible range: 1984-2024
           month = int, month of image. Possible range: 1-12
           day = int, day of image.
                 default: 1, otherwise possible range: 1-31
           reach = int, representing reach number. Number increases going upstream.
                   default: 1, applies for both validation and testing.
                   For training, the available range is 1-28 (included)
           img_res = int, image resolution (m).
                     default: 30 m.
           path = str, specifies directory where images are stored
                  default: r'data\satellite\original'
           num_images = int, sets the amount of images plotted
    
    Ouptuts: 
            none, series of images plotted
    '''
    if train_val_test is not None and reach is not None:
        directory =  os.path.join(path, collection) + '_' + train_val_test + '_r' + str(reach)
    else:
        directory = os.path.join(path, collection)

    # create and convert the starting date to a datetime object
    start_date = datetime.strptime(fr'{year}_{month}_{day}', '%Y_%m_%d')

    # initialize list to store image filenames
    image_files = []

    # iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.tif'):
            # convert filename date string to a datetime object
            if train_val_test is not None and reach is not None:
                file_date = datetime.strptime(filename.split(f'_{train_val_test}')[0], '%Y_%m_%d')
            else:
                file_date = datetime.strptime(filename.split(f'.')[0], '%Y_%m_%d')
            # check if the file date is equal to or after the start date
            if file_date >= start_date:
                image_files.append(filename)

    # sort image filenames based on date
    image_files.sort()

    # limit number of images to visualize
    image_files = image_files[:num_images]

    # calculate number of rows and columns for subplots
    num_cols = min(num_images, 5) # 5 images per row
    num_rows = (num_images - 1) // num_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 6.5*num_rows)) 

    # visualize the images in series
    for idx, filename in enumerate(image_files):
        if num_rows == 1:
            ax = axes[idx % num_cols]
        else:
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]

        img = mpimg.imread(os.path.join(directory, filename))
        ax.imshow(img, cmap='gray', vmin=0, vmax=2)
        ax.set_title(filename.split('.')[0].replace('_', '/'))
        # ax.set_title(filename.split('.')[0])  # display the date as the title
        
        shp = img.shape
        x_ticks = np.arange(0, shp[1], 300)
        y_ticks = np.arange(0, shp[0], 300)  

        # Convert x_ticks and y_ticks from pixels to meters
        x_tick_labels = [round(tick * img_res/1000, 2) for tick in x_ticks]  
        y_tick_labels = [round(tick * img_res/1000, 2) for tick in y_ticks]
  
        ax.set_xticks(x_ticks, x_tick_labels)
        ax.set_xlabel('Width (km)', fontsize=11)

    # hide any remaining empty subplots
    for idx in range(num_images, num_rows * num_cols):
        if num_rows == 1:
            ax = axes[idx % num_cols]
        else:
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]
        fig.delaxes(ax)

    # adjust title position depending on number of rows
    title_top = 0.98 + 0.004 * num_rows 

    fig.suptitle(f'Collection: {collection} - reach {reach} ({train_val_test})', fontsize=14, y=title_top)
    plt.tight_layout()
    plt.show()

    return None

def plot_input_images(train_val_test, reach_id, cmap='gray', vmin=0, vmax=2, img_res=60, 
                      dir_folders=r'data\satellite\dataset', collection=r'JRC_GSW1_4_MonthlyHistory', show=False):
    '''
    Plot all the images within a reach and use (training, validation and testing) that are used for the creation of the input dataset.
    
    Inputs:
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach_id = int, representing reach number. Number increases going upstream.
                      default: 1, applies for both validation and testing.
                   For training, the available range is 1-28 (included)
           cmap = str, key to set the visualization channels
                  default: 'gray'
           vmin = int, minimum value needed for visualization.
                  default: 0, can range from 0 to 255.  
           vmax = int, maximum value needed for visualization.
                  default: 2, can range from 0 to 255. 
           img_res = int, image resolution (m).
                     default: 60 m. 
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                       default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           show = bool, specifies whether single images are shown or not when calling teh `show_image_array` function.
                  default: False, if set to True images will be shown separately
    
    Output:
           none, plots all images in series 
    '''
    
    if collection != r'JRC_GSW1_4_MonthlyHistory':
        raise ValueError(fr'The collection set is {collection}, but this function only works for r"JRC_GSW1_4_MonthlyHistory"') 
    
    # get folder path
    folder = os.path.join(dir_folders, collection + fr'_{train_val_test}_r{reach_id}')
    # count number of images in the folder
    num_images = len(os.listdir(folder))

    num_cols = min(num_images, 5) # 5 images per row
    num_rows = (num_images - 1) // num_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 6.5*num_rows))

    # visualize the images in series
    for idx, filename in enumerate(os.listdir(folder)):

        year, month, day, _, _ = filename.split(f'_')

        if num_rows == 1:
            ax = axes[idx % num_cols]
        else:
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]

        img_path = get_path_images(dir_folders, collection, train_val_test, int(year), int(month), int(day), reach_id)
    
        image = show_image_array(img_path, cmap=cmap, vmin=vmin, vmax=vmax, show=show)
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{year}-{month}-{day}')
        
        # arrow to show flow direction
        ax.arrow(100, 125, 0, 200, width=10, facecolor='black', edgecolor='white') 
        
        shp = image.shape
        x_ticks = np.arange(0, shp[1]+1, 150)
        y_ticks = np.arange(0, shp[0]+1, 200)  

        # Convert x_ticks and y_ticks from pixels to meters
        x_tick_labels = [round(tick * img_res/1000, 2) for tick in x_ticks]  
        y_tick_labels = [round(tick * img_res/1000, 2) for tick in y_ticks]
  
        ax.set_xticks(x_ticks, x_tick_labels)
        ax.set_xlabel('Width (km)', fontsize=11)
        # ax.invert_yaxis() # can be confusing as flow is south-ward directed
        if col == 0:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels)
            ax.set_ylabel('Lentgh (km)', fontsize=11) 
        else:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([])
        
        # ax.axis('off')  # Hide axis

    # hide any remaining empty subplots
    for idx in range(num_images, num_rows * num_cols):
        if num_rows == 1:
            ax = axes[idx % num_cols]
        else:
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]
        fig.delaxes(ax)

    # adjust title position depending on number of rows
    title_top = 0.98 + 0.004 * num_rows 

    fig.suptitle(f'Reach {reach_id} {train_val_test}', fontsize=14, y=title_top)
    plt.tight_layout()
    plt.show()       

    return None