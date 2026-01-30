# Export Landsat Monthly Water History from Google Earth Engine

This repository contains a Jupyter/Colab notebook for exporting **Landsat-based monthly surface water history** imagery from **Google Earth Engine (GEE)**.  
The workflow is designed to batch-export images from the `JRC/GSW1_4/MonthlyHistory` collection with predefined river-system training rectangles (e.g. **Indus** and **Ghangara** systems). These rectangles can be made in QGis. 

The notebook is developed to increase the diversification in satellite imagery to train a FCNN model to predict braided river behaviour.  

---

## Overview

The notebook performs the following steps:

1. Authenticates and initialises the Google Earth Engine Python API  
2. Loads predefined rectangular regions of interest from GEE assets  
3. Filters the **JRC Global Surface Water – Monthly History** dataset by date  
4. Iterates over each rectangle and time step  
5. Submits export tasks to Google Drive  
6. Tracks submitted tasks to avoid duplicate exports

---

## Data Source

- **Dataset:** `JRC/GSW1_4/MonthlyHistory`  
- **Description:** Monthly global surface water classification derived from Landsat 5, 7, and 8  (the different rivers)
- **Temporal coverage:** 1984 – 2026  
- **Resolution:** 30 m  

Each exported image contains per-pixel water classification for a given month.

---

## Requirements

### Software
- Python 3.x
- Google Colab (recommended) or local Jupyter environment

### Python Packages
- `earthengine-api`
- `google-colab` (Colab only)
- `tqdm`
- `json`
- `datetime`
- `time`

### Accounts & Access
- Google Earth Engine account (approved)
- Access to the following GEE assets:
  - `projects/.../assets/Indus_training_rectangles_fixed`
  - `projects/.../assets/Ghangara_training_rectangles_fixed`

---

## Setup

### 1. Google Earth Engine Authentication
The notebook initializes Earth Engine using:

```python
ee.Initialize()