# African-Music-Genre-Classifier-
An end-to-end pipeline for African music genre classification. Includes an automated agent for mass dataset acquisition and a Jupyter Notebook for model training and analysis.

**African Music Genre Classifier 🌍🎵**
An end-to-end machine learning pipeline for the acquisition and classification of African musical genres.

**📖 Overview**
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

**🤖 Module 1: Data Acquisition Agent**

The agent is built to construct the dataset automatically by scraping audio sources based on genre queries.

**Key Features**
Automated Extraction: Downloads audio from YouTube using yt-dlp.

**Standardization:** Converts all audio to WAV format (16-bit PCM, Mono) with a configurable sample rate (default: 22050Hz).

**Auto-Organization:** Sorts files into genre-specific subfolders.

**Metadata Logging:** Automatically generates a CSV tracking filename, genre label, source URL, video title, and duration.

**Resilience:** Robust error handling ensures the script continues even if individual downloads fail.

**Output Structure**
After running the agent, your data directory will look like this:

**Plaintext**

african_music_dataset/
├── Afrobeats_Nigeria/
│   ├── dQw4w9WgXcQ.wav
│   └── ...
├── Highlife_Ghana/
│   └── ...
├── metadata.csv
└── error_log.txt
Metadata Format (metadata.csv)
**filename:** WAV file name (derived from YouTube ID to avoid special character issues).

**genre_label:** The specific search query used (e.g., "Highlife Ghana").

**youtube_url:** Direct link to the source.

**title:** Original video title.

**Duration:** Clip duration in seconds.

**🚀 Getting Started**
**1. Requirements**
Python 3.7+

ffmpeg (Critical: must be installed and in your system PATH)

Python Libraries: yt-dlp, pandas

**2. Installation**
**Step A:** Install FFmpeg

**Windows:** Download from ffmpeg.org and add to PATH.

**Linux:** sudo apt install ffmpeg

**Mac:** brew install ffmpeg

**Step B:** Install Python Dependencies

**Bash**

pip install -r requirements.txt
**3. Usage**
**Step A:** Configure Search Queries Edit search_queries.json with your target genres or specific search terms:

JSON

[
  "Afrobeats Nigeria",
  "Highlife Ghana",
  "Fuji music Nigeria",
  "Amapiano South Africa"
]
**Step B:** Run the Downloader

Bash

python download_agent.py
**Step C:** Configuration (Optional) You can adjust the following constants inside download_agent.py:

**MAX_RESULTS_PER_QUERY:** Number of tracks to fetch per genre (default: 50).

**SAMPLE_RATE:** Audio sample rate, e.g., 22050 or 44100 Hz (default: 22050).

**📓 Module 2: Classifier**
(Classification logic and notebook details to be added)

⚠️ Notes
Audio Format: Files are converted to mono to reduce dimensionality for the neural network.

Error Logging: Any failed downloads are logged to error_log.txt rather than crashing the script.

Suitability: The output WAV files are pre-processed specifically for ML/spectrogram analysis workflows.
