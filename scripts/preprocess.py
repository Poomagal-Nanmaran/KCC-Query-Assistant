import pandas as pd
import os
import json
from pathlib import Path

# Change this to your actual path
INPUT_CSV = "/app/data/raw/kcc_dataset.csv"
OUTPUT_JSON = "/app/data/preprocessed/kcc_cleaned_qa.json"

# Create output folders if not exist
Path("data/preprocessed").mkdir(parents=True, exist_ok=True)

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).strip().replace("\n", " ")
    return text

def preprocess_kcc_dataset():
    print(INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    print("File is read successfully")

    #required_columns = ['Query', 'Response']
    required_columns = ['QueryType','QueryText','KccAns']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    cleaned_data = []
    for _, row in df.iterrows():
        question = clean_text(row['QueryText'])
        answer = clean_text(row['KccAns'])

        if len(question) < 5 or len(answer) < 5:
            continue  # skip too short

        cleaned_data.append({
            "query": question,
            "response": answer,
            "metadata": {
                "QueryType": clean_text(row.get('QueryType', '')),
                "StateName": clean_text(row.get('StateName', '')),
                "DistrictName": clean_text(row.get('DistrictName', '')),
                "crop": clean_text(row.get('Crop', '')),
                #"language": clean_text(row.get('Language', '')),
                "date": clean_text(row.get('CreatedOn', ''))
            }
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(cleaned_data)} cleaned Q&A pairs to {OUTPUT_JSON}")

if __name__ == "__main__":
    preprocess_kcc_dataset()
