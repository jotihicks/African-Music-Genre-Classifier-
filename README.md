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
