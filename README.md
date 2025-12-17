# ecg-analytics
Using data science to reduce false positives in Atrial Flutter and Atrial Fibrillation.

**Project Background**

Atrial Fibrillation (AF) is an irregular rapid heartbeat that can lead to complications like stroke and heart failure (Mayo Clinic, n.d.). Atrial Flutter (AFL) is a rapid, regular atrial arrhythmia (~300 bpm). Symptoms: shortness of breath, dizziness, chest pain—or none (Mayo Clinic Staff, 2024).

Current detection includes automated algorithms, which is a rule-based QRS/RR analysis, deep learning that classifies raw 12- lead signals. These detections operate in real time and highly sensitive. However, false positives remain a practical barrier, which leads to anxiety and unnecessary treatment. Among ECGs with a computer diagnosis of AF/AFL, about 10% were incorrect on clinician review— demonstrating real-world false positives (Pan & Tompkins, 1985).

<img width="332" height="178" alt="image" src="https://github.com/user-attachments/assets/8147981e-2d69-4200-9743-2ad9a3b20d89" />

Clinically, reducing false positive filters out spurious positives and keeps clinician attention on the truly at-risk patients. Economically, it saves staff time and minimizes cost. The validator is lightweight, since it runs on CPU in real time. Therefore, gains come without added infrastructure cost (Pan & Tompkins, 1985).

**Research question**

“Can we reduce false positives for AF and AFL by adding a clinical validator—based on RR/HRV and flutter-band features?”

**Methodology**

2 Stages method:
Stage 1: Run HuBERT-ECG embedding with a Logistic head to predict AF and AFL.
Stage 2: Build a clinical-based validator on all positives, reject False Postives while keeping True Positives.
Stage 1:
Data Preprocessing and configuration based on HuBERT-ECG requirements (Coppola et al., 2025) with the following steps: 
-	Bandpass filter: [0.05, 47] Hz. 
-	Resampling at 100 Hz. 
-	Rescale signal to the [-1, 1] range. 
-	Use 5-second 12 leads.
Next, we used the HuBERT-ECG embeddings with the following process:
-	Loaded HuBERT-ECG via from_pretrained. 
-	Built embedding caches for train/val/test and generated embeddings. 
-	Trained a simple classifier head on embeddings. 
-	Set baseline at 0.50 threshold. 
-	Tuned thresholds on VAL: AF: recall ≥ 0.98 AFL: precision ≥ 0.30, recall ≥ 0.60.

I loaded the HuBERT-ECG backbone using `from_pretrained`, then generated and saved embedding caches for the train/validation/test splits. Using these embeddings as fixed features, I trained a lightweight classifier head on top as a strong baseline. I started with a default decision threshold of 0.50, then tuned class-specific thresholds on the validation set to match the target operating points: for AF, I enforced recall ≥ 0.98; for AFL, I targeted precision ≥ 0.30 and recall ≥ 0.60.


Stage 2:

For AF validator, the targeted recall is set to 99%. The features included are RR interval irregularity, P-wave absence, HRV metrics. Typical AF pattern is highly regular RR (↑ SDNN / ↑ CV / ↑ RMSSD, low regularity), absent/weak P-waves ( ↓ P-presence / ↓ P - energy, P:QRS < 1), and elevated HRV /entropy. The strategy is to keep very high recall threshold (≥ 0.99), we accept some false positives to ensure we catch all true cases.

For AFL Validator, a balanced approach is kept, with the precision ≥30%, recall ≥85%. The goal is to use AFL’s distinctive flutter wave patterns for better specificity. The features used are per-lead flutter wave analysis, harmonic components, atrial peak detection (Joglar et al., 2023). The strategy is to find the balance between Precision and Recall. The innovation is de-emphasizing RR features, since AFL can have regular or irregular RR features. 

**AF/AFL Feature Extraction (Validator Features)**

1) Detect heartbeats (QRS / R-peaks)

To locate heartbeats, I first enhance QRS complexes using a 5–20 Hz band-pass filter to keep the QRS band while suppressing baseline wander and high-frequency noise. This preprocessing makes R-peaks stand out as clear “humps” at each beat, enabling reliable QRS/R-peak detection.

2) Measure rhythm using RR intervals

After extracting R-peak timestamps, I compute RR intervals and derive rhythm/variability metrics:

RR mean: average RR interval (proxy for average heart rate)

SDNN: overall irregularity (typical: AF often > 100 ms, normal 30–60 ms)

CV (coefficient of variation): normalized variability (AF often > 0.15, normal < 0.10)

RMSSD: beat-to-beat variability (“chaos”)

Regularity index: 1 / (1 + CV) (higher = more regular rhythm)

3) P-wave check (to separate AF from sinus rhythm variability)

To avoid confusing AF with normal sinus rhythm + respiratory variation, I perform a P-wave presence/strength check. For each beat, I inspect a pre-QRS window (-200 ms to -50 ms), apply a 4–15 Hz band-pass filter, and measure P-wave energy.

Metrics:

Median P-energy: typical P-wave strength across beats

P-present fraction: % beats with a detectable P-wave (AF typically < 30%, normal > 70%)

P-to-QRS ratio: expected near 1:1 in normal sinus rhythm

4) AFL-specific features (flutter energy + best-lead selection)

For AFL detection, I focus on leads that commonly show flutter activity clearly: V1, II, V2, III, aVF. For each lead, I compute flutter-related spectral energy:

Flutter ratio: power in the flutter band (~4.2–5.8 Hz) relative to nearby baseline

Harmonic ratio: power in 8–12 Hz (2nd harmonic)

Combined flutter score: flutter_ratio + 0.5 × harmonic_ratio

I then select the lead with the highest combined score for that segment and use it as the representative AFL feature signal.

**Results:**

AF validator

<img width="589" height="148" alt="image" src="https://github.com/user-attachments/assets/3165d682-a6a7-4f00-8205-c701bb9f3afa" />


AF Result: 59% False Positives turned into True Negatives; 99% True Positives were kept.

AFL validator

<img width="595" height="122" alt="image" src="https://github.com/user-attachments/assets/11ed61ad-df7f-44b8-b659-636226639530" />

AFL Result: 31% False Positives turned into True Negatives; 86% True Positives were kept.

**Future work**

In short term, future work should strive for keeping 100% True Positives while still reduce significant number of False Positives. Cross-Dataset Validation & Generalization should also be included by testing the model with multiple data sets (Eg: PTB-XL) to assess its performance.

In long-term, the research should be extended to multi-arrhythmia and the explainability and interpretability should also be enhanced by having feature importance dashboard and confidence calibration to display probability scores with confidence intervals.
