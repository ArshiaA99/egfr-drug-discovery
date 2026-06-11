import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path


class MolecularFeaturizer:
    """Preprocesses chemical strings and transforms structural data into mathematical matrices."""

    def __init__(self, radius: int = 2, n_bits: int = 1024):
        self.radius = radius
        self.n_bits = n_bits

    def smiles_to_fingerprint(self, smiles: str) -> np.ndarray:
        """Generates a fixed-length Morgan fingerprint bit vector from a SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.n_bits
            )
            return np.array(fp)
        except Exception:
            return None

    def execute_pipeline(
        self, input_path: str, output_path: str = "egfr_processed_data.csv"
    ):
        """Runs the core transformation pipeline: pIC50 calculations and fingerprint featurization."""
        print(f"📋 Loading dataset from {input_path}...")
        df = pd.read_csv(input_path)

        # Step 1: Target Transformation
        print("🧮 Converting standard concentrations to pIC50 log scale...")
        df["pIC50"] = -np.log10(df["standard_value"] * 1e-9)
        df = df[(df["pIC50"] >= 2) & (df["pIC50"] <= 12)].copy()

        # Step 2: Fingerprint Generation
        print(
            f"🧩 Structural featurization into {self.n_bits}-bit Morgan Fingerprints..."
        )
        fingerprints = []
        valid_indices = []

        for idx, smiles in zip(df.index, df["canonical_smiles"]):
            fp = self.smiles_to_fingerprint(smiles)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)

        # Step 3: Matrix Structuring
        df_clean = df.loc[valid_indices].copy()
        fp_columns = [f"FP_{i}" for i in range(self.n_bits)]
        df_fp = pd.DataFrame(
            fingerprints, columns=fp_columns, index=df_clean.index
        )

        final_df = pd.concat(
            [
                df_clean[["molecule_chembl_id", "canonical_smiles", "pIC50"]],
                df_fp,
            ],
            axis=1,
        )

        print(f"✨ Featurization complete. Matrix dimensions: {final_df.shape}")
        final_df.to_csv(output_path, index=False)
        print(f"💾 Processed matrix saved to '{output_path}'")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    raw_data_path = str(BASE_DIR / "data" / "egfr_raw_data.csv")
    processed_data_path = str(BASE_DIR / "data" / "egfr_processed_data.csv")
    
    featurizer = MolecularFeaturizer(radius=2, n_bits=1024)
    featurizer.execute_pipeline(
        input_path=raw_data_path, output_path=processed_data_path
    )
