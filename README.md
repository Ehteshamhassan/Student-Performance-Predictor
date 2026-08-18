# Student-Performance-Predictor
AI student Performance Predictor..An interactive machine learning web application built with **Streamlit**, **Scikit-Learn**, and **Plotly** that predicts a student's final academic score based on academic history, study habits, and lifestyle factors.
#  AI Student Performance Predictor

An interactive machine learning web application built with **Streamlit**, **Scikit-Learn**, and **Plotly** that predicts a student's final academic score based on academic history, study habits, and lifestyle factors.

---

##  Features

* **Interactive Dual-Input Controls:** Synchronized sliders and numeric input fields for precise parameter tuning.
* **Random Forest Regressor:** Uses an ensemble ML model trained on multi-factor behavioral and academic data.
* **Real-time Performance Gauge:** Visualizes predicted final grades via an interactive Plotly gauge chart with dynamic status indicators.
* **Explainable AI (Feature Importance):** Breaks down model interpretability using horizontal bar charts to illustrate how each factor influences final scores.
* **Session State Management:** Includes state reset and auto-sync capabilities for responsive user interactions.

---

## Tech Stack

* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (`RandomForestRegressor`)
* **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Visualization:** [Plotly](https://plotly.com/python/)

---

##Input Metrics & Features

| Category | Input Feature | Description / Range |
| :--- | :--- | :--- |
| **Academic Profile** | Course / Major | Arts, Commerce, Mechanical Eng., Computer Science |
| | Previous Grade | 0% – 100% |
| **Study Habits** | Weekly Study Hours | 0 – 40 Hours |
| | Attendance | 0% – 100% |
| **Lifestyle Factors** | Sleep Duration | 2 – 12 Hours / Night |
| | Extracurriculars | Yes / No |
| | Part-Time Job | Yes / No |

---

## Getting Started

### 1. Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### 2. Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>
