import sqlite3
import pandas as pd
from sqlalchemy import create_engine

# Connect to SQLite database (or create it if it doesn't exist)
engine = create_engine('sqlite:///radiomics_data.db')
conn = sqlite3.connect('radiomics_data.db')

# Load data
covid_df = pd.read_csv('E:/Studies/Sem-5/SDP/Analysis Dashboard/extracted_features_Covid.csv')  # Replace with your actual file path
normal_df = pd.read_csv('E:/Studies/Sem-5/SDP/Analysis Dashboard/extracted_features_normal.csv')  # Replace with your actual file path

# Label each dataset
covid_df['Target'] = 'COVID'
normal_df['Target'] = 'Normal'

# Combine the datasets
data_df = pd.concat([covid_df, normal_df], ignore_index=True)

# Convert 'Target' column values to numeric labels
data_df['Target'] = data_df['Target'].apply(lambda x: 1 if x == 'COVID' else 0)

# Save the combined data into the database
data_df.to_sql('radiomic_features', conn, if_exists='replace', index=False)
