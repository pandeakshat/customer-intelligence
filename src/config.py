import os

# Define the path to your data folder
DATA_DIR = "data/sample"

# The 3 explicit files you mentioned
FILES = {
    "churn": os.path.join(DATA_DIR, "sample_churn.csv"),
    "segmentation": os.path.join(DATA_DIR, "sample_segmentation.csv"),
    "sentiment": os.path.join(DATA_DIR, "sample_sentiment.csv")
}