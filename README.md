# 🧬 Human EGFR Potency Prediction Engine

A production-ready, ligand-based virtual screening pipeline and FastAPI deployment framework designed to predict small-molecule bioactivity ($pIC_{50}$) against the **Human Epidermal Growth Factor Receptor (EGFR)**. 

This engine utilizes an enterprise-grade gradient boosting framework trained on validated biochemical assays to accelerate early-stage hit-to-lead oncology optimization.

---

## 📁 Repository Structure
```txt
├── data/
│   ├── egfr_raw_data.csv          <- Curated API pull from ChEMBL203
│   └── egfr_processed_data.csv    <- Featurized 1026-dimension matrix
├── models/
│   └── catboost_egfr_model.cbm    <- Serialized gradient boosted tree weights
├── src/
│   ├── init.py
│   ├── app.py                     <- FastAPI application with Tanimoto guardrails
│   ├── data_extraction.py         <- Asynchronous/paginated ChEMBL REST consumer
│   ├── pipeline.py                <- Scaffold-aware training loop
│   └── preprocess.py              <- Molecular descriptor featurization engine
```
---

## 🔬 Domain Biology & Target Architecture

### The Target: Human EGFR (ChEMBL203)
The Epidermal Growth Factor Receptor (EGFR; ErbB1; HER1) is a transmembrane glycoprotein and a member of the tyrosine kinase family. In many human epithelial malignancies—most notably **Non-Small Cell Lung Cancer (NSCLC)** and glioblastoma—EGFR is either overexpressed or undergoes oncogenic driver mutations. This causes the intracellular tyrosine kinase domain to continuously hydrolyze ATP, sending unchecked survival and proliferation signals to the cell nucleus.

### Mechanism of Action
The compounds modeled by this engine are **Small-Molecule Tyrosine Kinase Inhibitors (TKIs)**. They function via competitive inhibition: traveling into the intracellular pocket of the protein and physically blocking ATP from binding. Without ATP, the receptor cannot autophosphorylate, effectively shutting down the oncogenic signaling cascade.

---

## 📊 The Data & Featurization Pipeline

### 1. Data Ingestion & Biological Constraints
Raw bioactivity data is queried dynamically from the EMBL-EBI ChEMBL database using the target ID `CHEMBL203`. To eliminate experimental noise, the extraction engine enforces strict filtration:
* **Assay Uniformity:** Only compounds with explicitly defined $IC_{50}$ values measured in nanomolar (nM) concentrations are ingested.
* **Mathematical Standardization:**  Standard concentrations are converted to the logarithmic $pIC_{50}$ scale. The conversion is calculated as follows:
$pIC_{50} = -\log_{10}(\text{standard value} \times 10^{-9})$

  This linearizes the data, transforming highly skewed exponential concentrations into a balanced scale where a value of 9.0 represents 1 nM potency, 6.0 represents 1 µM potency, and values below 5.0 indicate weak or inactive structures.
### 2. High-Dimensional Molecular Featurization
Computers cannot interpret raw chemical strings (SMILES notation). The preprocessor transforms them into a unified **1,026-dimensional mathematical feature vector**:

| Feature Type | Dimensions | Method | Description |
| :--- | :--- | :--- | :--- |
| **Topological Fingerprints** | 1,024 bits | Morgan Fingerprints (Radius=2) | Generates a bit-vector representing local circular atom neighborhoods. It mimics the Extended Connectivity Fingerprint (ECFP4) standard used to capture structural fragments. |
| **Physical Descriptors** | 2 floats | Wildcard Calculators (RDKit) | Appends **Molecular Weight (MolWt)** to capture compound bulk/size and **Wildcard LogP (MolLogP)** to evaluate lipophilicity (how easily the molecule passes through cellular membranes). |

---

## ⚙️ Deep Dive: Generation of the 1,024 Fingerprint Columns

The 1,024 binary columns in `egfr_processed_data.csv` (ranging from `fingerprint_0` to `fingerprint_1023`) are built using a deterministic **Modified Morgan (ECFP4-equivalent) Circular Algorithm** via RDKit. 

Instead of treating a molecule as a simple text string, the algorithm maps it as a mathematical molecular graph where vertices are atoms and edges are chemical bonds.
```txt
[Atom: Carbon]  -- (Bond: Single) --  [Atom: Nitrogen]
         │                                      │
   (Initial Invariants: Atomic Number, Valence, Charge, Isotopes)
         │
  Iterative Expansion (Radius 1 ──> Radius 2 / 4 Bonds Max)
         │
Hash Generator Engine ──> Large Array of Unique Arbitrary Integers
         │
Modulo Folding Engine (Integer % 1024) ──> Dense 1024-Bit Allocation
```
### Step-by-Step Execution Vector:
1. **Initial Atom Invariants Assignment:** The preprocessor loops over every individual atom in the molecule. It assigns an initial integer code based on structural traits: atomic number, total number of attached hydrogens, valency, aromaticity, and formal charge.
2. **Iterative Neighborhood Exploration (Radius = 2):** * **Radius 0:** The algorithm records the atom's own properties.
   * **Radius 1:** The algorithm looks 1 bond away, pulling in the invariant codes of its direct neighbors and the traits of the connecting bonds. It aggregates these into a unified structural descriptor string.
   * **Radius 2 (ECFP4 Equivalent):** The algorithm sweeps outward to 2 bonds away (capturing paths up to 4 bonds across). This allows it to recognize larger functional rings and complex side-chains (e.g., the specific quinazoline ring structures common to EGFR inhibitors).
3. **Deterministic Hashing:** Every unique structural fragment discovered during the radius sweep is assigned a massive, arbitrary 32-bit integer ID using a hash function.
4. **Modulo Bit-Folding (The 1,024 Matrix Layout):** Because an infinite number of chemical sub-structures exist, the algorithm collapses these massive 32-bit integers into a fixed-length array of 1,024 positions using a modulo operation:
   $$\text{Bit Position} = \text{Fragment Hash} \pmod{1024}$$
   * If a fragment is present, the bit at that position flips to `1`.
   * If no fragment maps to that position, it remains `0`.
5. **Tabular Feature Output:** These 1,024 bits are unwrapped into individual standalone columns. This converts the molecule into a highly optimized sparse binary matrix that the CatBoost engine can split on instantly during training.

---

## 🧱 Scaffold-Aware Validation Strategy

A standard random train/test split fails completely in chemical machine learning. Because medicinal chemists synthesize structural "families" (dozens of molecules with the exact same core ring but minor atom variations), a random split results in severe **data leakage**. The model memorizes a chemical family in the training set and scores perfectly on near-identical clones in the test set, leading to catastrophic validation failure in clinical application.
```txt
RANDOM SPLIT (Flawed)                   BEMIS-MURCKO SPLIT (Production)
Train Set         Test Set               Train Set               Test Set
┌───────────┐     ┌───────────┐          ┌───────────┐           ┌───────────┐
│ ScaffoldA │ ──> │ ScaffoldA │          │ ScaffoldA │           │ ScaffoldB │
└───────────┘     └───────────┘          └───────────┘           └───────────┘
(Data Leakage / Memorization)            (True Out-of-Distribution Generalization)
```
To solve this, this pipeline implements a **Bemis-Murcko Scaffold Split**:
1. Every molecule’s side chains are stripped away, reducing the molecule to its fundamental ring architecture and linkers.
2. Compounds are grouped entirely by their unique core scaffolds.
3. Distinct scaffold groups are partitioned into training (80%) and validation (20%) sets. 

This guarantees that the model is evaluated on its ability to generalize to **entirely new chemical shapes** it has never encountered before. The achieved Test $R^2$ score of `~0.60` under this rigorous scaffold split confirms true predictive power.

---

## 🤖 Model Architecture & Analytics

The backend core relies on a **CatBoost Regressor** chosen for its unique advantages:
* **Symmetric Trees:** CatBoost builds balanced decision trees, providing incredibly fast inference speeds ($<5\text{ms}$) suited for live REST API endpoints.
* **Overfitting Resistance:** Built-in ordered boosting schemas naturally resist memorization, vital when learning sparse binary fingerprint matrices.

### Hyperparameter Configuration
```python
iterations=2000,
learning_rate=0.1,
depth=6,
loss_function="RMSE",
early_stopping_rounds=50
```
## 🚧 Applicability Domain & System Limitations
Ligand-based 2D QSAR (Quantitative Structure-Activity Relationship) models are highly specialized toolsets. A common phenomenon during deployment is seeing out-of-distribution chemicals or off-target compounds flag as active. This project incorporates structural safeguards to address these edge cases rather than using artificial hardcoded exceptions.

### The Out-of-Distribution (OOD) Phenomenon
Because public drug-discovery assays primarily document molecules that bind reasonably well, training data suffers from an inherent lack of "true negatives" (e.g., common household chemicals). Because decision tree models cannot extrapolate, arbitrary inputs default directly to the training median, resulting in false positives for completely benign compounds like Caffeine or Aspirin.

### The Imatinib Paradox
During testing, the leukemia drug Imatinib (Gleevec) returns a high predicted potency against EGFR. This behavior highlights the distinction between 2D topology and 3D biochemistry:
* Imatinib is a tyrosine kinase inhibitor designed for the BCR-ABL protein sequence.
* Because BCR-ABL and EGFR belong to the exact same kinase superfamily, they share highly similar ATP-binding pocket structures. Consequently, the core chemical scaffolds designed to fit them look almost identical on a flat 2D fingerprint map.
* The model correctly recognizes Imatinib as a high-potency kinase binder, but lacks the 3D spatial or spatial-conformation awareness required to detect that Imatinib does not fit perfectly into the specific shape of the EGFR pocket.

### Structural Guardrails (Tanimoto Filter)
To enforce an operational boundary, the FastAPI layer computes a live Tanimoto Coefficient vector across the entire query space on startup:
```python
similarities = DataStructs.BulkTanimotoSimilarity(query_fp, REFERENCE_FPS)
```
If an incoming string fails to meet a maximum similarity threshold of 0.48 against the verified training array, the engine flags the compound as outside the model's Applicability Domain and safely blocks the request before inference execution.

---

## 🚀 Local Deployment Integration
### Prerequisites
Ensure your environment contains the required C-compiled chemical informatics dependencies:
```bash
pip install fastapi uvicorn catboost rdkit numpy pandas scikit-learn matplotlib seaborn
```
### Execution
1. Extract Data: python src/data_extraction.py
2. Process Features: python src/preprocess.py
3. Train Engine: python src/pipeline.py
4. Launch Portal: python src/app.py

Navigate to http://127.0.0.1:8000/ to access the responsive lead optimization dashboard.

### Credits: Created by Arshia K ([Github: ArshiaA99](https://github.com/ArshiaA99))
