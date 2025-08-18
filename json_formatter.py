import pyarrow.parquet as pq
import pandas as pd
import json

if __name__ == "__main__":
    tbl = pq.read_table(
        "datasets/ChatDoctor-HealthCareMagic-100k/data/train-00000-of-00001-5e7cb295b9cff0bf.parquet"
    )
    df = tbl.to_pandas()

    records = df.to_dict(orient="records")  # List of dicts [{}, {}, ...]
    with open("datasets/HealthCareMagic-100k.json", "w") as f:

        json.dump(records, f, indent=2)
