import time
import pandas as pd
import requests
from pathlib import Path


class ChEMBLDataExtractor:
    """Handles querying and fetching bioactivity data directly from the ChEMBL REST API."""

    def __init__(self, target_id: str = "CHEMBL203"):
        self.target_id = target_id
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

    def fetch_raw_data(self, limit: int = 1000) -> list:
        """Queries the REST endpoint with pagination to retrieve raw compound records."""
        print(f"🚀 Connecting to ChEMBL REST API for Target: {self.target_id}")

        params = {
            "target_chembl_id": self.target_id,
            "standard_type": "IC50",
            "standard_units": "nM",
            "standard_relation": "=",
            "limit": limit,
            "offset": 0,
        }

        all_activities = []

        while True:
            print(f"📥 Fetching records starting at offset {params['offset']}...")
            try:
                response = requests.get(
                    self.base_url, params=params, timeout=30
                )

                if response.status_code != 200:
                    print(
                        f"❌ Server error {response.status_code}. Retrying in 5s..."
                    )
                    time.sleep(5)
                    continue

                data = response.json()
                activities = data.get("activities", [])

                if not activities:
                    break

                all_activities.extend(activities)
                params["offset"] += params["limit"]
                time.sleep(0.5)

            except Exception as e:
                print(f"⚠️ Connection glitch: {e}. Retrying in 5s...")
                time.sleep(5)
                continue

        print(
            f"📊 Extraction complete. Total records retrieved: {len(all_activities)}"
        )
        return all_activities

    def clean_records(self, raw_activities: list) -> pd.DataFrame:
        """Processes raw JSON responses into a structured, validated pandas DataFrame."""
        print("🧹 Cleaning data matrices and enforcing biological constraints...")

        records = [
            {
                "molecule_chembl_id": act.get("molecule_chembl_id"),
                "canonical_smiles": act.get("canonical_smiles"),
                "standard_value": act.get("standard_value"),
            }
            for act in raw_activities
        ]

        df = pd.DataFrame(records)

        # Drop invalid entries and enforce numeric types
        df = df.dropna(subset=["canonical_smiles", "standard_value"])
        df["standard_value"] = pd.to_numeric(
            df["standard_value"], errors="coerce"
        )
        df = df.dropna(subset=["standard_value"])

        # Enforce positive constraints and remove structural duplicates
        df = df[df["standard_value"] > 0]
        df = df.drop_duplicates(subset=["canonical_smiles"])

        print(f"✨ Filtering complete. Matrix shape: {df.shape[0]} compounds.")
        return df


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    output_path = str(BASE_DIR / "data" / "egfr_raw_data.csv")
    
    extractor = ChEMBLDataExtractor(target_id="CHEMBL203")
    raw_payload = extractor.fetch_raw_data()
    cleaned_df = extractor.clean_records(raw_payload)

    cleaned_df.to_csv(output_path, index=False)
    print(f"💾 Saved raw dataset artifact to '{output_path}'")
