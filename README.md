# JamUNet: Diversification of training data using new rivers.

<table>
  <tr>
    <td>
      <img src="./images/1994-01-25.png" width="1000" alt="Brahmaputra-Jamuna River">
    </td>
    <td style="vertical-align: top; padding-left: 20px;">
      <p style="font-size: 16px;">
        This repository is forked from the original repository by Antonio Magherini, which can be found in the 
        <a href="https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348">TU Delft repository</a>. 
        This updated version adopts changes/added files in the following folders:
      </p>
      <ul style="font-size: 14px; line-height: 1.6;">
        <li><strong>preprocessing</strong>: In <code>dataset_generation_modified.py</code> improved path handling and fault-tolerance for missing data.</li>
        <li><strong>preprocessing</strong>: In <code>satellite_analysis_pre.py</code>: Updated average loading logic.</li>
        <li><strong>preliminary</strong>: In <code>preliminary/edit_satellite_img.ipynb</code> the pipeline to add new river data was simplified to make it easier for future contributers to add new rivers.</li>
        <li><strong>postprocessing</strong>: In <code>plot_results.py</code>Enhanced visualization for batch-wise metrics.</li>
        <li><strong>models/st_unet</strong>: In <code>st_unet_3D</code> the new full 3D model was implemented (no time collapse in the bottleneck).</li>
      </ul>
    </td>
  </tr>
</table>

---

## Repository structure

The structure of this repository is the following:
- <code>data</code>, contains raw data (satellite images, river variables);
- <code>gee</code>, contains the scripts as <code>.js</code> files necessary for exporting the images from Google Earth Engine; (work in progress)
- <code>images</code>, contains the images shown in the thesis report and other documents;
- <code>model</code>, contains the modules and notebooks with the deep-learning model;
- <code>model_diversification</code>, contains experiments focused on diversifying the training set across different river systems;
- <code>model_extra_months</code>, contains investigations into using extended temporal sequences for prediction;
- <code>model_informativeness</code>, contains modules analyzing the information density and feature importance of the input data;
- <code>postprocessing</code>, contains the modules used for the data postprocessing;
- <code>preliminary</code>, contains the notebooks with the preliminary data analysis, satellite image visualization, preprocessing steps, and other examples;
- <code>preprocessing</code>, contains the modules used for the data preprocessing.

---

## Install dependencies

<code>braided.yml</code> is the environment file with all dependencies, needed to run the notebooks.

To activate the environment follow these steps:

- make sure to have the file <code>braided.yml</code> in your system (for Windows users, store it in <code>C:\Windows\System32</code>);
- open the anaconda prompt;
- run <code>conda env create -f braided.yml</code>;
- verify that the environment is correctly installed by running <code>conda env list</code> and checking the environment exists;
- activate the environment by running <code>conda activate braided_env_base</code>;
- deactivate the environment by running <code>conda deactivate</code>;