# Assessing Trends in Wildfire Severity in Catalonia Over the Last Decade: A Multi-Class Classification Approach Using U-Net

A deep learning project for wildfire severity assessment using satellite imagery, focusing on Catalunya (Catalonia), Spain. This project implements a U-Net semantic segmentation model to classify fire damage severity from post-fire satellite imagery.

## Data Sources

### 1. Catalunya Fire Polygons
Fire polygon data for Catalunya from 2014-2023 obtained from:
- **Source**: Generalitat de Catalunya - Departament d'Agricultura, Ramaderia, Pesca i Alimentació
- **URL**: https://agricultura.gencat.cat/ca/serveis/cartografia-sig/bases-cartografiques/boscos/incendis-forestals/incendis-forestals-format-shp/
- **Format**: Shapefiles containing fire boundaries, activation dates, and municipality information, from 2014 to 2023.

### 2. Labelled Dataset for Training
Pre-labeled wildfire severity dataset from:
- **Citation**: Luca Colomba, Alessandro Farasin, Simone Monaco, Salvatore Greco, Paolo Garza, Daniele Apiletti, Elena Baralis, and Tania Cerquitelli. 2022. A Dataset for Burned Area Delineation and Severity Estimation from Satellite Imagery. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (CIKM '22). Association for Computing Machinery, New York, NY, USA, 3893–3897. https://doi.org/10.1145/3511808.3557528
- **URL**: https://zenodo.org/records/6597139
- **Content**: ~14GB of Sentinel-1 GRD and Sentinel-2 L2A images with fire severity masks (5 classes: unburned to completely destroyed)

### 3. Satellite Imagery for Catalunya Fires
Post-fire satellite imagery retrieved via Sentinel Hub APIs:
- **Source**: Copernicus Data Space Ecosystem
- **API**: Sentinel Hub Processing API
- **Satellites**: Sentinel-1 GRD (IW mode, VV+VH polarization) and Sentinel-2 L2A
- **Temporal Coverage**: 2014-2023 (limited by satellite availability)
- **Criteria**: Images acquired ~1 month post-fire with <10% cloud coverage (for Sentinel-2 L2A data)

## Project Structure

```
DL_improc_final_project/
├── library/                          # Custom utility functions
│   ├── nn_model/                     # Neural network components
│   ├── preprocessing/                # Data preprocessing utilities
│   ├── utilities/                    # General utility functions
│   ├── visualizations/              # Visualization functions
│   └── eda/                         # Exploratory data analysis tools
├── catalunya_fire_data/             # Catalunya fire polygon data
│   └── clean_data/                  # Processed fire polygons
├── models/                          # Trained model weights
├── additional_content/              # Supplementary images and documentation
├── requirements.txt                 # Python dependencies
└── notebooks (described below)
```

## Notebooks Overview

### `1_1_catalunya_fires_cleaning.ipynb`
**Catalunya Fire Data Processing**
- Loads and merges fire polygon shapefiles from 2014-2023
- Standardizes column names and coordinate reference systems
- Creates an interactive map visualization of all fires
- Outputs: Clean unified shapefile for all Catalunya fires

### `1_2_sentinel_data_retrieval_fires.ipynb`
**Satellite Image Acquisition**
- Authenticates with Copernicus Data Space via OAuth2
- Implements automated image retrieval for all Catalunya fires from 2014-2023, if images are available within the temporal dimension
- Downloads Sentinel-1 GRD (4 bands) and Sentinel-2 L2A (13 bands) imagery
- Applies spatial buffering and temporal filtering (1 month post-fire)
- Creates georeferenced TIFF files with metadata
- Outputs: ~500+ satellite images covering Catalunya fires 2014-2023

### `2_preprocessing.ipynb`
**Data Preprocessing Pipeline**
- Tiles large satellite images into 256×256 pixel patches
- Processes both labelled training dataset and Catalunya inference dataset
- Handles overlapping tiles to ensure complete coverage
- Implements safety checks for dimension consistency
- Creates organized folder structure for tiled datasets
- Outputs: Training and inference-ready tiled datasets

### `3_eda.ipynb`
**Exploratory Data Analysis**
- Analyzes pixel value distributions in training masks (5 severity classes)
- Examines satellite image coverage quality and identifies defective tiles
- Visualizes Sentinel-2 band distributions across all 13 spectral channels
- Computes spectral indices (NDVI, NBR, etc.) and correlation analysis
- Removes corrupted/incomplete image tiles
- Provides insights for model training strategy

### `4_1_training_and_validation_S2.ipynb`
**U-Net Training - Sentinel-2**
- Implements U-Net architecture for semantic segmentation
- Trains on Sentinel-2 L2A data (13 bands) for fire severity classification
- Uses weighted cross-entropy loss to handle class imbalance
- Implements early stopping and best model checkpointing
- Applies data normalization and preprocessing
- Outputs: Trained U-Net model for Sentinel-2 data

### `4_2_training_and_validation_S1.ipynb`
**U-Net Training - Sentinel-1**
- Trains U-Net architecture on Sentinel-1 GRD data (4 bands)
- Adapts model for SAR imagery characteristics
- Uses same loss function and training strategies as Sentinel-2
- Handles different channel dimensions and data properties
- Outputs: Trained U-Net model for Sentinel-1 data

### `5_1_evaluation_on_test_S2.ipynb`
**Model Evaluation - Sentinel-2**
- Evaluates trained Sentinel-2 model on held-out test set
- Computes comprehensive metrics: accuracy, precision, recall, F1, IoU
- Generates confusion matrices for multi-class segmentation
- Creates precision-recall curves for optimal threshold selection
- Visualizes prediction samples and segmentation errors
- Provides detailed performance analysis

### `5_2_evaluation_on_test_S1.ipynb`
**Model Evaluation - Sentinel-1**
- Evaluates trained Sentinel-1 model performance
- Same evaluation methodology as Sentinel-2 for comparison
- Analyzes SAR-specific challenges and model capabilities
- Compares performance metrics between satellite types
- Documents model limitations and strengths

### `6_inference_catalunya_S2.ipynb`
**Catalunya Fire Severity Inference**
- Applies trained Sentinel-2 model to Catalunya fire imagery
- Processes all retrieved satellite images for severity assessment
- Generates severity maps for each fire event
- Creates visualizations of predicted fire damage
- Outputs: Fire severity assessments for Catalunya 2014-2023

## Key Features

- **Multi-modal satellite data**: Combines optical (Sentinel-2) and SAR (Sentinel-1) imagery
- **Automated data pipeline**: From raw satellite data to trained models
- **Class-balanced training**: Handles severe class imbalance in fire severity data, by upweighting the least frequent classes in the loss function
- **Comprehensive evaluation**: Multiple metrics and visualization tools
- **Real-world application**: Applied to actual Catalunya fire events

Collecting workspace information# DL_improc_final_project

A deep learning project for wildfire severity assessment using satellite imagery, focusing on Catalunya (Catalonia), Spain. This project implements a U-Net semantic segmentation model to classify fire damage severity from post-fire satellite imagery.

## Data Sources

### 1. Catalunya Fire Polygons
Fire polygon data for Catalunya from 2014-2023 obtained from:
- **Source**: Generalitat de Catalunya - Departament d'Agricultura, Ramaderia, Pesca i Alimentació
- **URL**: https://agricultura.gencat.cat/ca/serveis/cartografia-sig/bases-cartografiques/boscos/incendis-forestals/incendis-forestals-format-shp/
- **Format**: Shapefiles containing fire boundaries, activation dates, and municipality information

### 2. Labelled Dataset for Training
Pre-labeled wildfire severity dataset from:
- **Citation**: Luca Colomba, Alessandro Farasin, Simone Monaco, Salvatore Greco, Paolo Garza, Daniele Apiletti, Elena Baralis, and Tania Cerquitelli. 2022. A Dataset for Burned Area Delineation and Severity Estimation from Satellite Imagery. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (CIKM '22). DOI: https://doi.org/10.1145/3511808.3557528
- **URL**: https://zenodo.org/records/6597139
- **Content**: ~14GB of Sentinel-1 GRD and Sentinel-2 L2A images with fire severity masks (5 classes: unburned to completely destroyed)

### 3. Satellite Imagery for Catalunya Fires
Post-fire satellite imagery retrieved via Sentinel Hub APIs:
- **Source**: Copernicus Data Space Ecosystem
- **API**: Sentinel Hub Processing API
- **Satellites**: Sentinel-1 GRD (IW mode, VV+VH polarization) and Sentinel-2 L2A
- **Temporal Coverage**: 2014-2023 (limited by satellite availability)
- **Criteria**: Images acquired ~1 month post-fire with <10% cloud coverage

## Project Structure

```
DL_improc_final_project/
├── library/                          # Custom utility functions
│   ├── nn_model/                     # Neural network components
│   ├── preprocessing/                # Data preprocessing utilities
│   ├── utilities/                    # General utility functions
│   ├── visualizations/              # Visualization functions
│   └── eda/                         # Exploratory data analysis tools
├── catalunya_fire_data/             # Catalunya fire polygon data
│   └── clean_data/                  # Processed fire polygons
├── models/                          # Trained model weights
├── additional_content/              # Supplementary images and documentation
├── requirements.txt                 # Python dependencies
└── notebooks (described below)
```

## Key Features

- **Multi-modal satellite data**: Combines optical (Sentinel-2) and SAR (Sentinel-1) imagery
- **Automated data pipeline**: From raw satellite data to trained models
- **Class-balanced training**: Handles severe class imbalance in fire severity data
- **Comprehensive evaluation**: Multiple metrics and visualization tools
- **Real-world application**: Applied to actual Catalunya fire events

## Usage

1. **Setup**: Install dependencies and configure Copernicus Data Space credentials in .env, with the following credentials:
   ```
    COPERNICUS_CLIENT_ID='<client_key>'
    COPERNICUS_CLIENT_SECRET='<client_secret>'
   ```
2. **Data Preparation**: Run notebooks 1_1 → 1_2 → 2_preprocessing → 3_eda
3. **Model Training**: Execute 4_1 and/or 4_2 for Sentinel-2/Sentinel-1 models
4. **Evaluation**: Run 5_1 and/or 5_2 to assess model performance
5. **Inference**: Use 6_1 to apply models to Catalunya fire imagery

## Model Architecture

- **Architecture**: U-Net with encoder-decoder structure
- **Input**: Multi-spectral satellite imagery (4 or 13 channels)
- **Output**: 5-class fire severity segmentation
- **Training**: Weighted cross-entropy loss with early stopping

## Results

The project successfully demonstrates automated fire severity assessment using deep learning and satellite imagery, providing valuable tools for post-fire damage evaluation and emergency response planning.