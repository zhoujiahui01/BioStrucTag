# BioStrucTag
<img width="947" height="361" alt="image" src="https://github.com/user-attachments/assets/4ed53a57-0919-4a16-919e-b80a19de9f0f" />

# Introduction
The number of possible mutants increases exponentially as the number of mutable positions expands. To address this combinatorial explosion, we developed a complex structure-based machine learning framework, **BioStrucTag**, which integrates enzyme sequence features with voxelized three-dimensional representations of the active pocket binding environment. Trained on experimental stereoselectivity data, BioStrucTag enables efficient prioritization of promising variants for experimental validation through an iterative active-learning workflow.

**BioStrucTag is developed by QUB Huang Group : https://www.huanggroup.co.uk/**

**Find more AI tools in our enzyme design platform (LEAD-Zyme): https://www.greencatalysis.co.uk/lead-zyme**


# Required Dependencies
This package is tested with Python 3.10.16 and CUDA 12.8 on Ubuntu 22.04.
- Python 3.10
- PyTorch 2.6.0
- Transformers 4.51.3
- pyuul 0.4.2
- scikit-learn 1.6.1
- pandas 2.2.3
- numpy 1.26.4
- tensorflow 2.18.0
- joblib 1.4.2
- tqdm 4.67.1
  
# Usage
**Step 1. Preparation**

This step prepares the required sequence and structure inputs for downstream feature extraction and machine learning.
Prepare a CSV file, such as one in the **data/** folder, specifying mutation information and labels.
Generate protein sequences using the script:

```
python scripts/genesequences_from_csv.py \
    --input mutations.csv \
    --fasta wildtype.fasta \
    --output sequences.csv
```

All protein structures (.pdb format) should be placed in** data/structures/** . We recommend using AF3 to generate the script. The script is attached in the **example/** folder.
Generate active site structures using the script:

```
python scripts/activesiteextract.py \
  --center X Y Z
```

The extraction procedure consists of three steps:
1.1 Structural alignment

All input structures are aligned using PyMOL, and the aligned structures are saved to: **data/alignment/**

1.2 Active-site cropping

For each aligned structure, only atoms within a cubic box centered at a user-specified coordinate are retained.
The box size is fixed at 20 × 20 × 20 Å, corresponding to ±10 Å along each Cartesian axis. 

1.3 Output generation
The cropped active-site structures are saved in PDB format to: **data/activesite/**


**Step 2. Feature Generation**

**2a. Sequence Features (1D embeddings)**

Use the following script to generate ESM2-based embeddings from the sequences:

```
python scripts/generate_1Dembeddings.py 
```

**2b. Structural Features (3D embeddings)**
Generate voxelized 3D embeddings of the active site:

```
python scripts/generate_3Dembeddings.py
```

**Step 3. Model Training**
Train the model using the extracted sequence and structural feature:

```
python scripts/training.py \
    --label <LABEL_COLUMN>
```

<LABEL_COLUMN>: the column name in your CSV that contains the target property for regression.

**Step 4. Prediction on New Variants**
Prepare the input files like the operation in Step2. Predict properties for new sequences and structures:

```
python scripts/prediction.py \
    --input mutations-test.csv \
    --1d_dir outputs-test/1D_embeddings \
    --3d_dir outputs-test/3D_embeddings \
    --model models/rf_model.joblib
```

Predicted values will be saved to outputs-test/predictions.csv

**Directory Structure**

```
BioStrucTag/
├─ data/
│  ├─ structures/        # original structures
│  ├─ alignment/         # aligned structures 
│  ├─ activesite/        # cropped active-sites
│  └─ mutations.csv      # CSV with mutation info and labels
├─ outputs/
│  ├─ sequences.csv      # generated sequences
│  ├─ 1D_embeddings/     # ESM embeddings
│  └─ 3D_embeddings/     # voxel embeddings
├─ outputs-test/
│  ├─ 1D_embeddings/
│  ├─ 3D_embeddings/
│  └─ predictions.csv
├─ models/               # trained models
└─ scripts/              # all Python scripts
```

# Contact
Jiahui Zhou (jiahui.zhou@qub.ac.uk) Queen's University Belfast, UK

Meilan Huang (m.huang@qub.ac.uk) Queen's University Belfast, UK

