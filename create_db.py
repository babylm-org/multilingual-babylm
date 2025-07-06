import pandas as pd
import sqlite3

# Parameters
tsv_file = 'title.basics.tsv'
sqlite_db = 'imdb_mastersheet.db'
table_name = 'imdb'
chunksize = 10000  # Adjust based on your memory capacity

# Connect to SQLite (creates the file if it doesn't exist)
conn = sqlite3.connect(sqlite_db)

# Create table with tconst as primary key
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    tconst TEXT PRIMARY KEY,
    titleType TEXT,
    primaryTitle TEXT,
    originalTitle TEXT,
    isAdult INTEGER,
    startYear TEXT,
    endYear TEXT,
    runtimeMinutes TEXT,
    genres TEXT
)
"""
conn.execute(create_table_sql)

# Read and insert TSV in chunks
for i, chunk in enumerate(pd.read_csv(tsv_file, sep='\t', chunksize=chunksize)):
    # Write to SQLite; append to existing table
    chunk.to_sql(table_name, conn, if_exists='append', index=False)
    print(f'Inserted chunk {i + 1}')

conn.close()
print("Done loading TSV into SQLite.")