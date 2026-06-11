from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path


class CatBoostDrugModel:

    def __init__(self):
        """Initializes the pipeline model wrapper."""
        self.model = None

    def load_data(self, file_path):
        """Loads the processed biological dataset from a CSV file."""
        return pd.read_csv(file_path)

    def _add_physical_descriptors(self, smiles):
        """Internal helper to calculate molecular weight and LogP from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return pd.Series([Descriptors.MolWt(mol), Descriptors.MolLogP(mol)])
        return pd.Series([None, None])

    def engineer_features(self, data):
        """Calculates molecular properties and extracts the complete feature matrix."""
        print("🧪 Calculating Molecular Weight and LogP features...")
        data = data.copy()  # Avoid setting with copy warnings
        data[["MolWt", "LogP"]] = data["canonical_smiles"].apply(
            self._add_physical_descriptors
        )
        data = data.dropna(subset=["MolWt", "LogP"])
        return data

    def create_feature_label(self, data, target_name="pIC50"):
        """Separates the target vector from structural and physical descriptors."""
        fp_cols = [col for col in data.columns if col.startswith("FP_")]
        features = fp_cols + ["MolWt", "LogP"]

        X = data[features]
        y = data[target_name]
        return X, y

    def split_dataset(self, data, X, y, test_size=0.2):
        """Partitions data into training and validation subsets using Bemis-Murcko Scaffolds."""
        print("🧱 Applying Bemis-Murcko Scaffold Split...")
        scaffold_sets = defaultdict(list)

        # Group DataFrame indices by their unique ring architecture
        for idx, smiles in zip(data.index, data["canonical_smiles"]):
            try:
                mol = Chem.MolFromSmiles(smiles)
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                )
            except Exception:
                scaffold = ""
            scaffold_sets[scaffold].append(idx)

        # Sort scaffolds by family group size descending
        sorted_scaffold_sets = sorted(
            scaffold_sets.values(), key=lambda x: len(x), reverse=True
        )

        train_indices = []
        test_indices = []
        max_test_samples = int(len(data) * test_size)

        # Distribute whole groups to prevent structural data leakage
        for scaffold_indices in sorted_scaffold_sets:
            if len(test_indices) + len(scaffold_indices) <= max_test_samples:
                test_indices.extend(scaffold_indices)
            else:
                train_indices.extend(scaffold_indices)

        # Slice the feature matrices using aligned scaffold splits
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]

        print(f"📊 Split complete -> Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, X_test, y_test):
        """Configures and fits the CatBoost regressor with validation data monitoring."""
        print("🤖 Training CatBoost Regressor...")
        self.model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.1,
            depth=6,
            loss_function="RMSE",
            verbose=100,
            random_seed=42,
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
        )
        return self.model

    def evaluate_and_plot(self, y_test, y_pred):
        """Calculates validation metrics and displays regression performance."""
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("\n📊 Evaluating model performance...")
        print(f"📉 Test RMSE: {rmse:.4f}")
        print(f"📈 Test R² Score: {r2:.4f}")

        # Plot actual vs predicted values
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color="teal")
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            lw=2,
        )
        plt.xlabel("Actual pIC50")
        plt.ylabel("Predicted pIC50")
        plt.title("EGFR Potency Predictor: Actual vs Predicted")
        plt.tight_layout()
        plt.show()

    def save_model_artifact(self, filename="catboost_egfr_model.cbm"):
        """Persists the trained CatBoost configuration weights."""
        if self.model:
            self.model.save_model(filename)
            print(f"💾 Saved trained model artifact to '{filename}'")
        else:
            print("⚠️ No trained model instance found to save.")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    processed_data_path = str(BASE_DIR / "data" / "egfr_processed_data.csv")
    model_output_path = str(BASE_DIR / "models" / "catboost_egfr_model.cbm")
    
    # Initialize the OOP pipeline engine
    drug_pipeline = CatBoostDrugModel()

    # Pipeline execution flow
    data = drug_pipeline.load_data(processed_data_path)
    data = drug_pipeline.engineer_features(data)

    print(f"📊 Dataset stats: {data.shape[0]} compounds generated.")

    X, y = drug_pipeline.create_feature_label(data, target_name="pIC50")
    
    # Updated to pass 'data' so it splits cleanly by molecular architecture
    X_train, X_test, y_train, y_test = drug_pipeline.split_dataset(data, X, y)

    # Fit the encapsulated regressor model
    model_instance = drug_pipeline.train_model(X_train, y_train, X_test, y_test)

    # Perform inference and generate figures
    predictions = model_instance.predict(X_test)
    drug_pipeline.evaluate_and_plot(y_test, predictions)

    # Save model weights to file
    drug_pipeline.save_model_artifact(model_output_path)
