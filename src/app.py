import numpy as np
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# Initialize core FastAPI instance
app = FastAPI(
    title="EGFR Potency Engine",
    description="Production API for Human EGFR bioactivity predictions.",
    version="1.0.0"
)

# Load the optimized model artifact
MODEL_PATH = "catboost_egfr_model.cbm"
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

class PredictionRequest(BaseModel):
    smiles: str

def extract_features_from_smiles(smiles: str) -> np.ndarray:
    """Transforms a raw SMILES string into the exact 1026-feature vector expected by the model."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    # Generate 1024 structural features
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_array = list(np.array(fp))
    
    # Append the 2 optimized physical features
    mol_wt = Descriptors.MolWt(mol)
    log_p = Descriptors.MolLogP(mol)
    
    # Combine into unified 1026 feature row
    full_vector = fp_array + [mol_wt, log_p]
    return np.array(full_vector).reshape(1, -1)

@app.post("/predict")
def predict_potency(payload: PredictionRequest):
    features = extract_features_from_smiles(payload.smiles)
    if features is None:
        raise HTTPException(status_code=400, detail="Invalid chemical SMILES structure parsing string.")
        
    # Inference execution
    pIC50 = float(model.predict(features)[0])
    ic50_nm = 10 ** (9 - pIC50)
    
    # Categorize binding affinity
    if pIC50 >= 7.0:
        status = "Highly Potent Candidate (Strong Target Inactivation)"
        color = "text-emerald-400"
    elif pIC50 >= 5.0:
        status = "Moderately Active Entry"
        color = "text-amber-400"
    else:
        status = "Weak / Inactive Compound"
        color = "text-rose-400"
        
    return {
        "smiles": payload.smiles,
        "pIC50": round(pIC50, 4),
        "ic50_nm": round(ic50_nm, 2),
        "status": status,
        "color_class": color
    }

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves a professional, dark bioinformatics dashboard interface."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EGFR Lead Optimization Portal</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-950 text-slate-100 font-sans min-h-screen flex flex-col justify-between">
        
        <header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur px-6 py-4">
            <div class="max-w-6xl mx-auto flex items-center justify-between">
                <div class="flex items-center space-x-3">
                    <span class="text-2xl">🧬</span>
                    <h1 class="text-xl font-bold tracking-tight bg-gradient-to-r from-teal-400 to-cyan-400 bg-clip-text text-transparent">
                        EGFR Cancer Drug Discovery Portal
                    </h1>
                </div>
                <span class="px-3 py-1 text-xs font-semibold text-teal-400 bg-teal-950/50 border border-teal-800 rounded-full">
                    CatBoost Engine Online
                </span>
            </div>
        </header>

        <main class="flex-grow max-w-4xl w-full mx-auto p-6 flex flex-col justify-center">
            <div class="bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl space-y-6">
                <div>
                    <h2 class="text-lg font-medium text-slate-200">Molecular Potency Inference</h2>
                    <p class="text-sm text-slate-400 mt-1">Input a canonical SMILES string below to calculate estimated affinity metrics against Human EGFR.</p>
                </div>

                <div class="space-y-2">
                    <label class="text-xs font-semibold tracking-wider uppercase text-slate-400">SMILES Notation</label>
                    <div class="flex gap-3">
                        <input type="text" id="smilesInput" placeholder="e.g., CN1C=NC2=C1C(=O)N(C(=O)N2C)C" 
                               class="flex-grow bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-200 focus:outline-none focus:border-teal-500 transition-colors font-mono text-sm">
                        <button onclick="runInference()" id="btnText" class="bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-600 hover:to-cyan-600 px-6 py-3 rounded-xl font-medium text-slate-950 transition-all flex items-center justify-center min-w-[120px]">
                            Analyze
                        </button>
                    </div>
                </div>

                <div id="resultCard" class="hidden border border-slate-800 bg-slate-950/50 rounded-xl p-6 space-y-4 transition-all">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="bg-slate-900/60 p-4 rounded-lg border border-slate-800/40">
                            <span class="text-xs text-slate-400 block uppercase tracking-wider">Predictive Score (pIC50)</span>
                            <span id="pic50Res" class="text-3xl font-bold text-teal-400 font-mono block mt-1">--</span>
                        </div>
                        <div class="bg-slate-900/60 p-4 rounded-lg border border-slate-800/40">
                            <span class="text-xs text-slate-400 block uppercase tracking-wider">Equiv concentration (IC50)</span>
                            <span id="ic50Res" class="text-3xl font-bold text-cyan-400 font-mono block mt-1">--</span>
                        </div>
                    </div>
                    <div class="pt-2 border-t border-slate-800/60 flex items-center justify-between">
                        <span class="text-xs text-slate-400 uppercase tracking-wider">Oncology Screening Profile:</span>
                        <span id="statusRes" class="font-bold text-sm">--</span>
                    </div>
                </div>
            </div>
        </main>

        <footer class="border-t border-slate-900 bg-slate-950 text-center py-4 text-xs text-slate-500">
            Computational Biotechnology Research Pipeline • Target Architecture: CHEMBL203
        </footer>

        <script>
            async function runInference() {
                const smiles = document.getElementById("smilesInput").value.trim();
                const resultCard = document.getElementById("resultCard");
                const btnText = document.getElementById("btnText");
                
                if(!smiles) return alert("Please present a valid SMILES string.");
                
                btnText.innerText = "Processing...";
                btnText.disabled = true;

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ smiles: smiles })
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok) {
                        alert(data.detail || "Transformation structural parsing error.");
                        return;
                    }

                    document.getElementById("pic50Res").innerText = data.pIC50;
                    document.getElementById("ic50Res").innerText = data.ic50_nm + " nM";
                    
                    const targetStatus = document.getElementById("statusRes");
                    targetStatus.innerText = data.status;
                    targetStatus.className = "font-bold text-sm " + data.color_class;

                    resultCard.classList.remove("hidden");
                } catch(err) {
                    alert("Inference connection failed.");
                } finally {
                    btnText.innerText = "Analyze";
                    btnText.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting local web server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
