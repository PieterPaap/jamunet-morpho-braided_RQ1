## Data

This folder contains the data used for this project. 

<code>The Zip file</code> Contains the entire repo. The repo is quite large, uploading it without the zip was very difficult.

Repo explaination:

Data augmentation:
The data was edited using code cells not currently in the repo, but the edited data is still there.
What was done is that the .tiff file orginisation was edited to 1 year, with 12 months, this is so that that it would be easier with the dataloader, my pc had great difficulties without this.

All this data can be found in "data" -> "satellite" -> "preprocessed_PIETER" -> "any folder" -> ".npy" files.
<br>
<br>
<br>

The edited code: <br>
<br>
BE WARY, there are alot of lines with # in de .py files scroll down to see the active parts.
<br>

In "preprocessing" the "dataset_generation" as edited a little bit to make de dataloader work, because it did not like my changes.

In "model" -> "lazydata.py" this is a seperate .py file for the dataloading, loading the data takes alot of time, so i implemented "num_workers" to load the data faster. Each worker starts a python instance to load the data. So i made this a seperate small file to keep the overhead small.

In "model" -> "st_unet" -> "st_unet" this is the location of the small "SIMPLE" 3D model where time collapses relatively quickly. BE WARY, there are alot of lines with # scroll down to see the active part.

In "model" -> "st_unet" -> "st_unet_complex" this is the location of the "COMPLEX" 3D model with full time. BE WARY, there are alot of lines with # scroll down to see the active part.

In "model" -> "train_eval" & "train_eval_copy" these two files do the training and validation for the simple and the complex model, there used to be some differences but i removed them. The inside code is the exact same. BE WARY, there are alot of lines with # scroll down to see the active part.

In "model" -> "Unet3D_spatial_...._SIMPLE" this is the notebook where the simple model is run from.

In "model" -> "Unet3D_spatial_...._COMPLEX" this is the notebook where the complex model is run from.

In "postprocessing" -> "metrics" in this file some of the code was edited so that the metric would work with the new code, the old metrics did not like my model changes.