# SmartRetina
Integrated Web Interface for Classification of Age-related Macular Degeneration (AMD), Diabetic Retinopathy (DR), Glaucoma
This project is an integrated deep learning–based web application for the automatic detection of three major retinal diseases using fundus images:

- Age-related Macular Degeneration (ARMD)
- Diabetic Retinopathy (DR)
- Glaucoma

The system uses trained CNN and hybrid deep learning models and offers an easy-to-use interface built with Flask. Upload a fundus image and get instant predictions across all three disease classes.

---

**Project Structure**


├── app.py # Main Flask app
├── predict_armd.py # Prediction function for ARMD
├── predict_dr.py # Prediction function for DR
├── predict_gla.py # Prediction function for Glaucoma
├── models/
│ ├── armd_model.h5 # Trained ARMD model
│ ├── dr_effnetb0.pt # Trained DR model (EfficientNet-B0)
│ ├── glaucoma_model.pth # Trained Glaucoma model
├── requirements.txt # List of Python dependencies


**Models Used**
ARMD: Custom U-Net + CNN-LSTM Hybrid Model 
DR: EfficientNet-B0 based classifier 
Glaucoma: EfficientNet-B0 based binary classifier 


** Applications**
Rural & Occupational Health Screening
Mass AI-powered Retinal Disease Diagnosis
Real-time Fundus Image Analysis

