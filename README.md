# 🩺 Heart Disease Risk AI

**Real-time cardiovascular risk prediction achieving 97% sensitivity with <200ms inference — engineered for emergency triage, preventive cardiology, and enterprise-scale decision support.**

*By Ridwan Oladipo, MD | Clinical AI Architect*

---

[![🎬 UI Demo](https://img.shields.io/badge/🎬_UI_Demo-Live-blue?style=flat-square)](https://huggingface.co/spaces/dr-ridwanoladipo/cardio-ai)
[![🔗 API Demo](https://img.shields.io/badge/🔗_API_Demo-Live-green?style=flat-square)](https://huggingface.co/spaces/dr-ridwanoladipo/cardio-ai-api)  
[![🚀 Production (AWS ECS Fargate)](https://img.shields.io/badge/🚀_Production-cardio.mednexai.com-f59e0b?style=flat-square)](#-deployment-options)    
[![GitHub](https://img.shields.io/badge/Code-Repository-00aa00?style=flat&logo=github&logoColor=white)](https://github.com/dr-ridwanoladipo/cardio-ai)

> **Clinically aligned Heart-AI system built with physician-led modeling, guideline-integrated decision support, and full AWS production MLOps.**

---
## 🎯 Executive Summary
Heart disease remains the leading global cause of death, where rapid risk stratification directly impacts survival in emergency and preventive settings.  
This system delivers **<200ms cardiovascular risk prediction** with **97% sensitivity**, **SHAP explainability**, and guideline-aligned recommendations—deployed on **AWS Fargate** and ready for hospital EHR integration, telemedicine workflows, and enterprise clinical decision support.

---
## 📊 Performance at a Glance

| Metric | Value | Clinical Meaning |
|:--|:--|:--|
| **Sensitivity** | **0.97** | Captures nearly all true heart disease cases |
| **Specificity** | **0.71** | Balanced false-positive rate for safe triage |
| **ROC-AUC** | **0.91** | Strong overall discriminative power |
| **PPV** | **0.80** | High reliability when predicting disease |
| **NPV** | **0.95** | Very low risk of missing true negatives |

![Model Performance](outputs/model_evaluation.png)

> Clinically tuned for **safety-first medicine** — prioritizing early detection and minimizing missed disease.

---
## 🌐 Deployment Options
- **Live Demos**: Instant access via HuggingFace (UI + API)
- **Production (On-Demand)**: Fully deployed on AWS ECS Fargate at *cardio.mednexai.com* — **available by request**  
>⚡ **AWS Production**: cardio.mednexai.com — CI/CD-enabled, <10 minutes cold-start (cost-optimized)

---
## 💼 Business Impact
- **Risk Stratification Automation**: Replaces manual ASCVD/Framingham scoring, saving 5–10 minutes per patient across thousands of annual encounters
- **Preventive Cardiology**: Identifies high-risk patients before acute events, reducing avoidable ED admissions and downstream costs
- **EHR & Telemedicine Ready**: API-first design enables plug-and-play deployment into Epic, Cerner, and remote-care platforms
---

## 🏗️ Medical Workflow Architecture

```mermaid
flowchart LR

%% ------------------------
%% Clinical Workflow
%% ------------------------
subgraph Clinical_Workflow
A[Patient Inputs\nAge, Symptoms, Vitals, ECG] --> B[Streamlit Clinical UI]
B --> C[FastAPI Backend\nValidation and Safety Checks]
end

%% ------------------------
%% AI Pipeline
%% ------------------------
subgraph AI_Pipeline
C --> D[XGBoost Model\nFast Inference]
C --> E[SHAP Explainer\nAttribution]
D --> F[Risk Prediction]
E --> G[Feature Contributions]
end

%% flow back to UI
F --> B
G --> B

%% ------------------------
%% AWS Infrastructure
%% ------------------------
subgraph AWS_Production
H[ECR Registry] --> I[ECS Fargate Task]
I --> J[Application Load Balancer]
K[Route 53 Domain] --> J
J --> C
end
```

---

## 🎬 Interactive Features

### **Clinical Interface**
- Real-time cardiovascular risk scoring (<200ms)
- Color-coded risk gauge with confidence analysis  
- SHAP waterfall + clinical explanations  
- Guideline-aligned recommendations (AHA/ACC + NCEP)  
- Population percentile visualization  

### **API Integration**
Production-grade FastAPI endpoints with full documentation:

```bash
curl -X POST "https://dr-ridwanoladipo-cardio-ai-api.hf.space/predict" \
     -H "Content-Type: application/json" \
     -d '{"age":63,"sex":1,"cp":0,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

# Interactive docs:
https://dr-ridwanoladipo-cardio-ai-api.hf.space/docs
```

---
## 🎨 Visual Showcase

### **Risk Assessment Dashboard**
**High-risk patient example:**  
![Risk Gauge](outputs/risk-assessment.png)

**Low-risk patient example:**  
![Low Risk](outputs/risk-assessment2.png)

### **SHAP Explainability**
![SHAP Waterfall](outputs/shap-explanation.png)

### **Clinical Decision Support**
![Clinical Support](outputs/clinical-support.png)

___

## 🏗️ Technical Architecture & MLOps
- **Model**: XGBoost optimized with Optuna (50+ trials) + SMOTE class balancing for minority-case sensitivity  
- **Dataset**: Cleveland Heart Disease (303 patients, UCI) - standard cardiology ML benchmark with clinically validated features
- **Feature Engineering**: Age stratification, AHA/ACC blood-pressure categories, NCEP cholesterol risk levels, ECG-related interactions  
- **Robustness**: 97% sensitivity maintained across demographic subgroups with clinical plausibility checks  

**Production Stack**: XGBoost • FastAPI • Streamlit • AWS ECS Fargate • Docker • GitHub Actions • CloudWatch  
**CI/CD**: Automated ECS deployments with health checks, rollback, and zero-downtime (~5 min git-push → production)  
**Explainability**: SHAP top-feature attribution • Guideline-aligned clinical interpretation • Population percentile benchmarking  

---

## 🧪 Clinical Validation & Compliance
- Stratified train-val-test split with cross-validation, class-balanced evaluation, and demographic subgroup analysis
- All medical decisions should be made in consultation with qualified healthcare providers

---

## 👨‍⚕️ About the Developer
**Ridwan Oladipo, MD** — Clinical AI Architect  
Physician + MLOps engineer delivering production-ready medical AI across cardiology, radiology, and multimodal diagnostics (7+ deployed systems).

### **Connect & Collaborate**
> [![🌐 portfolio](https://img.shields.io/badge/🌐_portfolio-mednexai.com-1e3c72?style=flat-square)](https://mednexai.com)
[![linkedin](https://img.shields.io/badge/linkedin-connect-0077b5?style=flat-square&logo=linkedin)](https://linkedin.com/in/drridwanoladipoai)
[![email](https://img.shields.io/badge/email-contact-d14836?style=flat-square&logo=gmail)](mailto:dr.ridwan.oladipo@gmail.com)

- **Open to**: Senior Medical Data Scientist • Clinical AI Architect • Applied ML / MLOps Engineer  
- **Collaboration**: Hospitals, research teams, startups, and engineering groups building real-world medical or AI-driven products


