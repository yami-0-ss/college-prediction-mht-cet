import streamlit as st
import pandas as pd
import numpy as np
import catboost as cb
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="MHT-CET College Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🎓 MHT-CET College & Branch Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict your best-fit engineering college and branch using your trained CatBoost classification model.</p>', unsafe_allow_html=True)

# Load CatBoost Model
@st.cache_resource
def load_model():
    model = cb.CatBoostClassifier()
    model.load_model("MHT_CET_CatBoost_Model.cbm")
    return model

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}. Please ensure 'MHT_CET_CatBoost_Model.cbm' is in the project directory.")
    st.stop()

# Sidebar Input Fields
st.sidebar.header("🔍 Enter Your Academic Details")

rank = st.sidebar.number_input("Enter MHT-CET Rank", min_value=1, max_value=500000, value=15000, step=100)
percentile = st.sidebar.number_input("Enter MHT-CET Percentile", min_value=0.0, max_value=100.0, value=92.5, step=0.1)
marks = st.sidebar.number_input("Enter MHT-CET Marks", min_value=0.0, max_value=200.0, value=110.0, step=0.5)
category = st.sidebar.selectbox("Select Category", ["OPEN", "OBC", "SC", "ST", "VJ/DT", "NT-B", "NT-C", "NT-D", "EWS"])

# Format input DataFrame (adjust feature names if your training dataset used different identifiers)
input_data = pd.DataFrame({
    'Rank': [rank],
    'MHT_CET_Percentile': [percentile],
    'MHT_CET_Marks': [marks],
    'Category': [category]
})

# Prediction Trigger
if st.sidebar.button("Predict Best College"):
    with st.spinner("Analyzing profile and calculating probabilities..."):
        try:
            # Get multi-class probabilities
            probs = model.predict_proba(input_data)[0]
            classes = model.classes_
            
            # Construct a sorted DataFrame of predictions
            pred_df = pd.DataFrame({
                'College & Branch': classes,
                'Probability': probs
            }).sort_values(by='Probability', ascending=False).reset_index(drop=True)
            
            top_prediction = pred_df.iloc[0]['College & Branch']
            top_prob = pred_df.iloc[0]['Probability'] * 100
            
            # Display Results Layout
            st.success("Prediction Complete!")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🏆 Top Recommended Allocation")
                st.info(f"**{top_prediction}**")
                st.metric(label="Confidence Score", value=f"{top_prob:.2f}%")
                
            with col2:
                st.subheader("📊 Top 5 Probabilities Distribution")
                top_5 = pred_df.head(5)
                fig = px.bar(
                    top_5, 
                    x='Probability', 
                    y='College & Branch', 
                    orientation='h',
                    text_auto='.1%',
                    color='Probability',
                    color_continuousScale='blues'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
            # Full Results Table Expandable
            with st.expander("📋 View All College Options & Probabilities"):
                st.dataframe(pred_df, use_container_width=True)
                
        except Exception as ex:
            st.error(f"Prediction failed: {ex}")
            st.info("Ensure that the input columns match the exact features expected by your CatBoost model.")

st.markdown("---")
st.markdown("🚀 *Powered by Streamlit and CatBoost*")
