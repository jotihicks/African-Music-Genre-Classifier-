# African-Music-Genre-Classifier-
An end-to-end pipeline for African music genre classification. Includes an automated agent for mass dataset acquisition and a Jupyter Notebook for model training and analysis.

## 📖 Overview
This repository contains the codebase for a deep learning project aimed at classifying distinct African music genres (e.g., Afrobeat, Highlife, Amapiano, Soukous, Chimurenga). The project consists of two main modules:

**Data Acquisition Agent:** An automated bot designed for mass downloading and organizing audio samples to construct a robust dataset.

**Classification Model:** A Jupyter Notebook containing the preprocessing logic, feature extraction (MFCCs/Spectrograms), and model training pipeline.

**🎯 Target Genres**
The model will be trained to distinguish between the following classes:

Afrobeat

Afrobeats

Amapiano

Highlife

Makosa

Kwaito

Soukous

Juju

Fuji

Congolese Rumba

## 🤖 Module 1: Data Acquisition Agent

The agent is built to construct the dataset automatically by scraping audio sources based on genre queries.

## Features

- Downloads audio from YouTube using `yt-dlp`
- Converts to WAV format (16-bit PCM, configurable sample rate)
- Organizes by genre in separate folders
- Logs metadata (filename, genre, URL, title, duration) to CSV
- Robust error handling - continues on failures
- Configurable number of results per query

## Requirements

- Python 3.7+
- ffmpeg (must be installed and in PATH)
- yt-dlp
- pandas

## Installation

1. Install ffmpeg:
   - Windows: Download from https://ffmpeg.org/download.html
   - Linux: `sudo apt install ffmpeg`
   - Mac: `brew install ffmpeg`

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Edit `search_queries.json` with your genres/search terms:
```json
[
  "Afrobeats Nigeria",
  "Highlife Ghana",
  "Fuji music Nigeria"
]
```

2. Run the downloader:
```bash
python download_agent.py
```

3. Configure settings in the script:
   - `MAX_RESULTS_PER_QUERY`: Number of tracks per genre (default: 50)
   - `SAMPLE_RATE`: Audio sample rate - 22050 or 44100 Hz (default: 22050)

## Output Structure

```
african_music_dataset/
├── Afrobeats_Nigeria/
│   ├── dQw4w9WgXcQ.wav
│   └── ...
├── Highlife_Ghana/
│   └── ...
├── metadata.csv
└── error_log.txt
```

## Metadata CSV

Contains:
- `filename`: WAV file name
- `genre_label`: Genre/query label
- `youtube_url`: Source URL
- `title`: Video title
- `duration`: Duration in seconds

## Notes

- Files are named by YouTube video ID to avoid special character issues
- Audio is converted to mono, 16-bit PCM WAV
- Script continues on errors and logs them to `error_log.txt`
- Suitable for ML/spectrogram analysis workflows

## 🚀 Notebook Workflow & Methodology

The classification pipeline is divided into four sequential Jupyter Notebooks. They are designed to take raw `.wav` audio files and transform them into both tabular and image datasets for traditional and deep learning models.

### 1. `Feature_Extraction.ipynb` (Tabular Data Generation)
**Goal:** Extract numerical acoustic features from the raw audio files to create a structured dataset for classical machine learning.
* **Process:** The notebook mounts to Google Drive and traverses the genre-specific folders containing the 1,000 trimmed `.wav` files. 
* **Extraction:** Using `librosa`, it analyzes the time-domain and frequency-domain of each track to calculate the **mean** and **variance** of:
  * Zero Crossing Rate (ZCR)
  * Root Mean Square (RMS) Energy
  * Estimated Tempo (BPM)
  * Spectral Centroid
  * 20 Mel-Frequency Cepstral Coefficients (MFCCs)
* **Output:** The extracted values are compiled into a master pandas DataFrame and exported as `features_extracted.csv` (resulting in a dataset with 90+ numerical feature columns).

### 2. `Spectrogram_Creation.ipynb` (Image Data Generation)
**Goal:** Convert audio signals into visual representations (images) so they can be processed by a Convolutional Neural Network (CNN).
* **Process:** The notebook loops through the audio dataset and applies a Short-Time Fourier Transform (STFT) using `librosa`. 
* **Transformation:** It converts the audio amplitude to the decibel scale (`librosa.amplitude_to_db`) to generate Mel-spectrograms.
* **Formatting:** To ensure the CNN only learns from the audio data and not the plot borders, the notebook strips the axes, labels, and white space (`plt.axis('off')`). 
* **Output:** Saves high-resolution `.png` spectrogram images into new genre-specific directories.

### 3. `african_music_genre_classification.ipynb` (Classical ML Pipeline)
**Goal:** Perform Exploratory Data Analysis (EDA) and train traditional machine learning models on the extracted CSV features.
* **EDA & Visualization:** Begins by visualizing a sample audio waveform (e.g., Congolese Rumba) and its properties. It then loads the CSV and uses `seaborn` to generate box plots and scatter plots, mapping how features like MFCCs and Spectral Centroids vary across the 10 African genres.
* **Preprocessing:** The numerical features are normalized using `StandardScaler` to ensure models converge properly. The data is split into an 80/20 train-test split (848 training samples, 213 testing samples).
* **Model Training:** Several classifiers from `scikit-learn` are trained and compared, including:
  * Support Vector Machine (SVM) with RBF Kernel
  * Logistic Regression
  * Random Forest & Gradient Boosting
  * Stochastic Gradient Descent (Perceptron)
  * Naive Bayes & Decision Trees
* **Results:** The **SVM** achieved the highest accuracy at **72.3%**. The notebook concludes by plotting a normalized Confusion Matrix to visualize the model's precision and recall across overlapping genres (like Afrobeat vs. Afrobeats).

### 4. `CNN_AFMGC.ipynb` (Deep Learning Pipeline)
**Goal:** Build and train a custom Convolutional Neural Network (CNN) to classify the spectrogram images.
* **Data Loading:** Imports the `.png` spectrograms generated in Notebook 2 using TensorFlow/Keras utilities.
* **Architecture:** Defines a deep learning model consisting of:
  * Multiple `Conv2D` layers (e.g., 32, 64, 128, 512 filters) for extracting spatial patterns from the spectrogram frequencies.
  * `MaxPooling2D` layers to reduce dimensionality and computational load.
  * `Dropout` layers to prevent the model from over-fitting on the training data.
  * A `Flatten` layer leading into a fully-connected `Dense` network, ending with a 10-node softmax output for genre prediction.
* **Training & Evaluation:** The model is compiled with the `Adam` optimizer and trained over 30 epochs. It achieves a robust **training accuracy of ~98.3%** and a **validation accuracy of ~89.6%**, proving that deep learning on spectrograms captures the nuances of African music genres significantly better than classical ML on numerical features.
