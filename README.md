# MedPrioNet
MedPrioNet: A Neural Network Guided by
Medical Prior Knowledge for Diagnosing Focal
Liver Lesions
**<p align="center">Abstract</p>**
Accurate identification of liver lesions is essential for optimizing patient management and enhancing the early diagnosis of liver cancer. Contrast-enhanced
ultrasound (CEUS), a non-invasive imaging technique offering high spatial and temporal resolution, is essential for diagnosing focal liver lesions (FLLs). However, most existing diagnostic methods fail to fully utilize the specific information contained in CEUS, thereby compromising the accurate identification of the lesion characteristics.
Therefore, we propose a neural network model guided by medical prior knowledge (MedPrioNet) for the automatic diagnosis of FLLs. This model fully incorporates lesion characteristics and offers better diagnostic interpretability. Firstly, we construct a Multi-scale Feature Extraction Network (MFE-Net) to effectively extract features from single frame lesion images. On this basis, we further propose a Dynamic Perfusion Temporal Attention (DPTA) module in
combination with a key-frame guided diagnostic strategy to model temporal dependencies across CEUS sequences. This design enables effective extraction of diagnostic cues by jointly leveraging temporal perfusion dynamics and key frame differences, thereby effectively capturing the blood
flow dynamics and key pathological information of liver lesions. The proposed model achieves diagnostic accuracies of 90.19% and 79.87% on the FLLs dataset and the breast lesion dataset, respectively. These results demonstrate the potential of the model to serve as a more accurate computer-aided diagnostic tool.



# Requirements:
● Python 3.8