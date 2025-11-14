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
flowchart TB

%% ============================
%%      CLINICAL WORKFLOW
%% ============================
subgraph "🩺 Clinical Workflow (Primary Layer)"

A[🧑‍⚕️ Patient Clinical Inputs\nAge • Symptoms • Vitals • ECG-related features] 
--> B[🖥️ Streamlit Clinical UI\n(Triage Interface • Risk Gauge • XAI View)]
B --> C[🧠 FastAPI Backend\nValidation • Routing • Safety Checks]

end

%% ============================
%%         AI PIPELINE
%% ============================
subgraph "🤖 AI Pipeline (Inference Layer)"

C --> D[XGBoost Model\n<200ms Inference]
C --> E[SHAP Explainer\nFeature Attribution]
D --> F[Risk Prediction\nLow • Moderate • High]
E --> G[Top Feature Contributions\nGuideline-aligned Insight]

end

%% Back to UI (closing the loop)
F --> B
G --> B

%% ============================
%%     AWS PRODUCTION LAYER
%% ============================
subgraph "☁️ AWS Production (Infrastructure Layer)"

H[ECR Container Registry] --> I[ECS Fargate Task\nAuto-scaling • Secure • Isolated]
I --> J[Application Load Balancer\nHTTPS • Health Checks]
J --> C
K[Route 53\ndomain: cardio.mednexai.com] --> J

end

%% ============================
%%        VISUAL STYLING
%% ============================

%% Clinical workflow colors
style A fill:#e0f7fa,stroke:#006064,color:#004d40,stroke-width:1.5px
style B fill:#ede7f6,stroke:#5e35b1,color:#311b92,stroke-width:1.5px
style C fill:#f3e5f5,stroke:#8e24aa,color:#4a148c,stroke-width:1.5px

%% AI pipeline colors
style D fill:#fff3e0,stroke:#ef6c00,color:#e65100,stroke-width:1.5px
style E fill:#f1f8e9,stroke:#33691e,color:#1b5e20,stroke-width:1.5px
style F fill:#ffe0b2,stroke:#f57c00,color:#e65100
style G fill:#dcedc8,stroke:#558b2f,color:#33691e

%% AWS infrastructure colors
style H fill:#e3f2fd,stroke:#1e88e5,color:#0d47a1,stroke-width:1.5px
style I fill:#bbdefb,stroke:#1976d2,color:#0d47a1,stroke-width:1.5px
style J fill:#90caf9,stroke:#1565c0,color:#0d47a1,stroke-width:1.5px
style K fill:#b3e5fc,stroke:#0288d1,color:#01579b,stroke-width:1.5px
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


