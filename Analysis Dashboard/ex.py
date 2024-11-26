import sqlite3
import pandas as pd
from sqlalchemy import create_engine

conn = sqlite3.connect('radiomics_data.db')

# Verify the data stored in the database
query = "SELECT * FROM radiomic_features LIMIT 5"
sample_data = pd.read_sql(query, conn)
print(sample_data)
