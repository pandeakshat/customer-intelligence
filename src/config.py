# src/config.py
import os

# Points to the folder where you dropped your CSVs
DATA_DIR = os.path.join("data", "sample")

FILES = {
    "churn": os.path.join(DATA_DIR, "sample_churn.csv"),
    "segmentation": os.path.join(DATA_DIR, "sample_segmentation.csv"),
    "sentiment": os.path.join(DATA_DIR, "sample_sentiment.csv")
}