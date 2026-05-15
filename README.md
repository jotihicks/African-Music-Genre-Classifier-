# African-Music-Genre-Classifier(AFMGC)-
An end-to-end pipeline for African music genre classification. Includes an automated agent for mass dataset acquisition and a Jupyter Notebook for model training and analysis.

## 📖 Overview
This repository contains the codebase for a deep learning project aimed at classifying distinct African music genres (e.g., Afrobeat, Highlife, Amapiano, Soukous, Chimurenga). The project consists of two main modules:


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

### 4. `Convolutional Neural Networks on AFMGC` (Deep Learning Pipeline)
**Goal:** Build and train a custom Convolutional Neural Network (CNN) to classify the spectrogram images.
* **Data Loading:** Imports the `.png` spectrograms generated in Notebook 2 using TensorFlow/Keras utilities.
* **Architecture:** Defines a deep learning model consisting of:
  * Multiple `Conv2D` layers (e.g., 32, 64, 128, 512 filters) for extracting spatial patterns from the spectrogram frequencies.
  * `MaxPooling2D` layers to reduce dimensionality and computational load.
  * `Dropout` layers to prevent the model from over-fitting on the training data.
  * A `Flatten` layer leading into a fully-connected `Dense` network, ending with a 10-node softmax output for genre prediction.
* **Training & Evaluation:** The model is compiled with the `Adam` optimizer and trained over 30 epochs. It achieves a robust **training accuracy of ~98.3%** and a **validation accuracy of ~89.6%**, proving that deep learning on spectrograms captures the nuances of African music genres significantly better than classical ML on numerical features.


## 5. ⚠️ Known Limitations & Next Steps

This project represents a foundational pipeline. The results reported above are honest *for the evaluation protocol used*, but several methodological choices warrant transparency — and they directly motivate the research direction I plan to pursue at the PhD level.

### 1. Album/Artist Leakage in the Train/Test Split

The current 80/20 split is random, which means tracks by the same artist likely appear in both training and test sets. This is a well-documented pitfall in music genre classification (Sturm, 2013; Flexer & Schnitzer, 2010), and it tends to inflate deep-learning performance more than classical-ML performance because CNNs can learn artist-specific timbral, vocal, and production cues rather than genre-level features.

The 17-percentage-point gap between SVM (72.3%) and CNN (89.6%) is consistent with this phenomenon and almost certainly overstates the CNN's true generalisation to unseen artists.

**Next step:** Re-evaluate with *artist-disjoint* and *album-disjoint* splits, and report the corrected numbers alongside the originals. I expect the CNN's honest validation accuracy to drop into the 70–80% range and the SVM–CNN gap to narrow substantially.

### 2. Genre Label Validation

Labels were derived from YouTube search queries (e.g. "Highlife Ghana", "Afrobeats Nigeria"). These queries are noisy: a search for "Afrobeats Nigeria" returns a mix of Afrobeats, Afrobeat, Highlife, and other adjacent genres, and uploader tagging on YouTube is not ethnomusicologically validated.

The genuine ambiguity between certain genre pairs — Afrobeat vs Afrobeats, Highlife vs Afrobeats, Soukous vs Congolese Rumba — is itself musicologically meaningful, but it cannot be properly characterised without expert annotation.

**Next step:** Collaborate with ethnomusicologists at the University of Lagos and partner institutions to validate a curated subset of the dataset, ideally with multi-annotator inter-rater agreement scores. This work is core to the broader data-sovereignty direction of the project.

### 3. Dataset Size, Quality, and Provenance

1,061 tracks is small by modern MIR standards — contemporary audio foundation models are trained on hundreds of thousands of hours. YouTube provenance also means variable audio quality across the dataset: different codecs, bitrates, and recording conditions per track.

**Next step:** Expand the dataset through partnership with African music archives and through field recordings, addressing provenance and consent explicitly. Where direct expansion is constrained, leverage transfer learning from large pretrained audio models (CLAP, MERT, AST) rather than training CNNs from scratch — a more appropriate paradigm for low-resource musical traditions.

### 4. Feature Choices Are Western-Music-Centric

MFCCs and standard mel-spectrograms were originally developed for Western popular and classical music. African musical traditions involve characteristics — polyrhythmic structures, microtonality, non-12-TET tuning systems, characteristic ornamentation — that may not be optimally represented by these features.

**Next step:** Explore rhythm-aware representations (longer-context onset detection, beat-synchronous features) and self-supervised pre-training on African music itself, so that learnt representations are shaped by the data rather than by inherited Western feature design.

### 5. Evaluation Methodology

The current report relies on a single train/test accuracy figure per model. A more rigorous protocol would include macro-F1 (given class imbalance), per-class precision and recall, k-fold cross-validation, and confidence intervals from repeated runs.

**Next step:** Adopt the standard MIR evaluation protocol — k-fold CV, macro-F1, per-class metrics with confidence intervals, and appropriate statistical tests for model comparison.

### 6. Ethical and Data Sovereignty Considerations

The current dataset was scraped from YouTube without explicit artist consent, and the genre taxonomy was imposed externally rather than negotiated with the communities whose music is being classified. For a project framed around inclusion and representation, this is a contradiction that the project must eventually address.

**Next step:** Move towards a collaboratively-curated dataset aligned with the CARE principles for indigenous data governance (Collective benefit, Authority to control, Responsibility, Ethics; Carroll et al., 2020), and engage with artists directly where possible.

---

These limitations are not afterthoughts — they form the agenda for the PhD research I am preparing to undertake. The current pipeline establishes that the problem is *technically tractable*; the next phase is to make it *methodologically rigorous and ethically grounded*.

### References

- Carroll, S. R., Garba, I., Figueroa-Rodríguez, O. L., Holbrook, J., Lovett, R., Materechera, S., et al. (2020). The CARE principles for indigenous data governance. *Data Science Journal*, 19(43), pp. 1–12.
- Flexer, A. and Schnitzer, D. (2010). Effects of album and artist filters in audio similarity computed for very large music databases. *Computer Music Journal*, 34(3), pp. 20–28.
- Sturm, B. L. (2013). The GTZAN dataset: its contents, its faults, their effects on evaluation, and its future use. *Journal of New Music Research*, 43(2), pp. 147–172.
