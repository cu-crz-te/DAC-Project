# Gaming Community NLP & Topic Modeling Project

This repository contains an end-to-end Natural Language Processing project conducted in collaboration with the NTU Data Analytics Club a leading game publisher. The goal was to analyze large-scale player discussions from multiple online communities, uncover major themes, and extract insights that could support strategies for improving player engagement.

*Note:*
*1. The source code contains comments in Traditional Chinese as this project was originally developed in a Mandarin-speaking business context.*
*2. Some result files and visualizations are not included in this repository due to confidential information related to the industry partner. Only non-sensitive examples and structural outputs are provided for demonstration purposes.*

---

## 🔍 Project Overview

We collected more than twenty thousand player comments from several gaming forums and social media platforms. The project applied topic modeling, sentiment analysis, and LLM-assisted tagging to identify key concerns, emotional reactions, and behavioral patterns among players.

This repository includes the complete reproducible workflow, from data preprocessing to modeling and insight generation.

---

## 📂 Repository Structure

```
DAC-Project/
│── notebooks/
│ ├── 00_environment_setup.ipynb
│ ├── 01_config_settings.ipynb
│ ├── 02_data_preprocessing.ipynb
│ ├── 03_bertopic_training.ipynb
│ ├── 04_llm_settings.ipynb
│ ├── 05_topic_evaluator.ipynb
│ ├── 06_model_training.ipynb
│ ├── 07_model_loading_postprocessing.ipynb
│ ├── 08_topic_keywords.ipynb
│ └── 09_model_evaluation.ipynb
│── src/
│ ├── preprocessing.py
│ ├── modeling.py
│ └── evaluation.py
└── README.md
```

---

## 🛠 Methods & Techniques

### **Natural Language Processing**
- Text cleaning and normalization  
- Tokenization & stopword removal  
- n-gram exploration  

### **Topic Modeling**
- BERTopic  
- UMAP dimensionality reduction  
- HDBSCAN clustering  
- Topic merging, re-labeling, and consolidation  

### **LLM-Assisted Tagging**
- Few-shot prompting  
- Chain-of-thought reasoning  
- Topic and sentiment auto-labeling  
- Manual validation to ensure accuracy  

### **Model Evaluation**
- Topic coherence evaluation  
- Interpreting high-probability terms  
- Sentiment distribution analysis  

---

## 🎯 Key Findings

- Consolidated discussions into **six major topic clusters**
- Achieved **over 80 percent accuracy** with LLM-assisted topic and sentiment tagging  
- Identified emotional triggers and user pain points  
- Revealed sentiment shifts tied to character balancing and event updates  
- Derived insights that supported community engagement strategy proposals  


