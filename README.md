# college-prediction-mht-cet
# 🎓 MHT-CET College & Branch Predictor

An interactive, web-based machine learning application built with **Streamlit** and **CatBoost** to help students predict their best-fit engineering colleges and branches based on their MHT-CET exam metrics (Rank, Percentile, Marks, and Category).

🌐 **Live App:** [View Deployed App](https://college-prediction-mht-cet-cdmeqbetglrqkr2xsxvisd.streamlit.app/)

---

## 🚀 Features

* **Real-time Predictions:** Instantly computes multi-class classification probabilities across top Maharashtra engineering institutions.
* **Interactive UI:** Clean sidebar widgets for entering rank, percentile, marks, and category.
* **Data Visualization:** Uses **Plotly** to render horizontal bar charts displaying the top 5 college options and confidence distributions.
* **Comprehensive Breakdown:** Expandable data table view to inspect the full list of college predictions and probabilities.

---

## 🛠️ Tech Stack

* **Python 3.8+**
* **Streamlit** (Web Framework)
* **CatBoost** (Machine Learning Model Inference)
* **Plotly Express** (Interactive Charts)
* **Pandas & NumPy** (Data Manipulation)

---

## 📁 Project Structure

```text
├── MHT_CET_CatBoost_Model.cbm    # Trained CatBoost multi-class model binary
├── app.py                        # Main Streamlit application script
├── requirements.txt              # Python package dependencies
└── README.md                     # Project documentation
