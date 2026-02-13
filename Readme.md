# 📄 Reproduction of  
## Unsupervised Anomaly Detection Improves Imitation Learning for Autonomous Racing

---

## 📌 Project Goal

This project aims to reproduce the anomaly detection pipeline and experimental results presented in:

**Unsupervised Anomaly Detection Improves Imitation Learning for Autonomous Racing**

Specifically, we aim to reproduce:

- PCC reconstruction curves  
- Median-filtered PCC anomaly detection curves  

Comparison across:

- Proposed method (**Lrec + Lrefer**)  
- Ablation baseline (**Lrec only**)  
- (Optional) SOTA baseline (**GCL**)  

---

## 📌 Target Figure to Reproduce

The target is to reproduce PCC-based reconstruction quality plots including:

### Raw PCC Curves
For anomaly types:
- Raindrop  
- Wall hit  
- Plastic obstruction  

### Median Filtered PCC Curves
Using temporal median filter:
Window size = 100 frames (~5 seconds driving time)


---

## 📌 High-Level Pipeline

The full pipeline consists of two stages:

### Stage 1 — Unsupervised Anomaly Detection

Train Convolutional Autoencoder (CAE) to:

- Reconstruct normal images accurately  
- Reconstruct abnormal images poorly  

Loss function:

L = Lrec + λ Lrefer


---

### Stage 2 — PCC-Based Anomaly Scoring

Compute per-frame anomaly score using:

- Pearson Correlation Coefficient (PCC)  
- Temporal Median Filtering  
- Adaptive Threshold Detection  

---

## 📌 Dataset

Dataset structure:

clean
raindrop
plastic
hitwall
foggy
...


Image shape:
(N, 224, 224, 3)


Training:
clean images only


Testing:
clean + anomaly category


---

## 📌 Implementation Plan

### Phase 1 — Data Loading
Tasks:
- Implement dataset loader for `.pkl` dataset  
- Convert to PyTorch tensors  
- Normalize to [0, 1]  

Deliverable:
dataset.py


---

### Phase 2 — CAE Model Training

Train two models:

#### Model A — Proposed Method
Loss:
L = Lrec + λ Lrefer


#### Model B — Ablation Baseline
Loss:
L = Lrec only


Deliverables:
train_ours.py
train_ablation.py


---

### Phase 3 — Reference Latent Construction

Per training epoch:
- Sample M reference frames  
- Encode into latent space  
- Store reference latent set  

---

### Phase 4 — PCC Computation

For each frame:
PCC(original_image, reconstructed_image)


Deliverable:
compute_pcc.py


---

### Phase 5 — Median Filtering

Apply temporal median filter:
Window size = 100


Deliverable:
median_filter.py


---

### Phase 6 — Plotting

Reproduce:
- Raw PCC curves  
- Median filtered PCC curves  

Deliverable:
plot_pcc_curves.py


---

## 📌 Key Mathematical Components

### Reconstruction Loss
MSE(x, x_recon)


---

### Latent Reference Loss
Nearest-neighbor latent distance MSE


---

### PCC Score
Measures structural similarity:
Range = [-1, 1]


---

### Anomaly Threshold
δ = median(PCC) - 0.05


---

## 📌 Expected Reproduction Timeline

### Step 1
- CAE training complete  
- PCC computation verified  
- Raindrop PCC curve generated  

---

### Step 2
- Median filtering implemented  
- Multi-anomaly evaluation complete  

---

### Step 3
- Full figure reproduction completed  
- Performance comparison validated  

---

## 📌 Potential Challenges

- Efficient nearest-neighbor latent search  
- Stable training of latent reference loss  
- Correct PCC normalization  
- Applying median filter along time axis  

---

## 📌 Evaluation Criteria

Reproduction success will be evaluated by:

- Trend consistency with paper plots  
- Separation between clean and anomaly PCC distributions  
- Stability after median filtering  

Exact numeric matching is not required.

---

## 📌 Tools & Libraries

Recommended stack:
PyTorch
NumPy
SciPy
Matplotlib


---

## 📌 Suggested Repository Structure

project/
├ models/
├ losses/
├ training/
├ eval/
├ plotting/
├ dataset/
└ README.md


---

## 📌 Final Objective

Demonstrate that:

Unsupervised anomaly detection
→ Dataset cleaning
→ Improved imitation learning robustness


---

## 📌 Notes

This reproduction focuses on validating anomaly detection behavior trends rather than exact numeric replication.

---
