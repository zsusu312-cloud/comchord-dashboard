#  ComChord Leadership Dashboard

### Author: **Zhu Xun**
### Institution: *Nanyang Technological University (NTU), Singapore*  
### Deployment: [🌐 Live Demo on Streamlit](https://comchord-dashboard-leadership.streamlit.app)

---

##  Overview

**ComChord Leadership Dashboard** is an NLP-powered analytics tool designed to evaluate **leadership communication effectiveness** across multiple 1-on-1 meetings.  
It combines text mining, sentiment analysis, and topic modeling to uncover how communication tone, clarity, and engagement evolve over time.

This project was independently designed and implemented for **ComChord’s data analytics interview assessment**, focusing on data preprocessing, visualization, and business insight storytelling.

---

##  Features

- **Leadership KPI Dashboard**
  - Visualizes metrics such as *Talk Ratio*, *Empathy Index*, and *Team Positivity*.
- **Sentiment Evolution**
  - Tracks how team morale changes across meetings.
- **Engagement Behavior**
  - Measures action- and question-driven dialogue patterns.
- **Key Topic Extraction**
  - Identifies recurring discussion areas using TF-IDF analysis.
- **Communication Clarity**
  - Quantifies understanding and initiative between manager and team.
- **PDF Report Export**
  - Auto-generates structured leadership summaries in PDF format.

---

##  Data & Pipeline

| Step | Description |
|------|--------------|
| **1. Input** | Meeting transcripts from `input_meetings/` (Word files) |
| **2. Processing** | NLP preprocessing using NLTK, spaCy, TextBlob |
| **3. Output** | Cleaned analytics files in `output_data/` |
| **4. Visualization** | Streamlit app (`app_streamlit.py`) |
| **5. Export** | Automated PDF report via ReportLab |

> *Note: The preprocessing file (`final.py`) is disabled on Streamlit Cloud to simplify deployment.*

---

##  Tech Stack

- **Python**: pandas · numpy · plotly · streamlit  
- **NLP Libraries**: nltk · spacy · textblob  
- **Visualization**: plotly.express  
- **Report Generation**: reportlab  
- **Version Control**: GitHub + Streamlit Cloud Deployment

---

## 📁 Repository Structure

# comchord-dashboard
