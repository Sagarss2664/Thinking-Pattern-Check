import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tempfile
import shutil

# Import utility functions
from utils.image_processing import process_image, extract_letters
from utils.prediction import MPredictor

# Page configuration
st.set_page_config(
    page_title="Handwriting Analysis System",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load CSS
try:
    local_css("static/styles.css")
except:
    pass

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'm_images' not in st.session_state:
    st.session_state.m_images = []
if 'results' not in st.session_state:
    st.session_state.results = []
if 'normalized_scores' not in st.session_state:
    st.session_state.normalized_scores = {}

# App title and description
st.title("✍️ Handwriting Thinking Pattern Analysis")
st.markdown("""
    This system analyzes handwritten text (specifically the letter 'M') to determine 
    the writer's thinking patterns. Upload an image of handwritten text to begin analysis.
""")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Upload Handwritten Script")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, PNG, HEIC)",
        type=["jpg", "jpeg", "png", "heic"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
    st.markdown("---")
    st.markdown("### How it works:")
    st.markdown("""
    1. Upload an image of handwritten text
    2. System detects all 'M' letters in the text
    3. Each 'M' is analyzed for thinking patterns
    4. Results show pattern distribution
    """)
    st.markdown("---")
    st.markdown("### Thinking Patterns:")
    st.markdown("""
    - **Cumulative**: Methodical, step-by-step thinking
    - **Investigative**: Questioning, exploratory thinking
    - **Comprehensive**: Big-picture, holistic thinking
    - **Analytical**: Logical, detail-oriented thinking
    """)

# Main content area
if st.session_state.uploaded_file is None:
    st.info("Please upload a handwritten script image using the sidebar.")
    st.image("https://via.placeholder.com/600x400?text=Upload+Handwritten+Script", 
             caption="Example of handwritten text to analyze")
else:
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file to temp location
        temp_file_path = os.path.join(temp_dir, st.session_state.uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(st.session_state.uploaded_file.getbuffer())
        
        # Process the image
        with st.spinner("Processing uploaded image..."):
            try:
                img_cv = process_image(temp_file_path)
                
                # Extract M letters
                output_folder = os.path.join(temp_dir, "extracted_ms")
                os.makedirs(output_folder, exist_ok=True)
                st.session_state.m_images = extract_letters(img_cv, output_folder)
                
                if not st.session_state.m_images:
                    st.error("No 'M's were found in the script. Please upload a different image.")
                    st.stop()
                
                st.success(f"Found {len(st.session_state.m_images)} 'M's in the script!")
                
                # Show sample of extracted M's
                st.subheader("Sample of Extracted 'M's")
                cols = st.columns(min(5, len(st.session_state.m_images)))
                for i, col in enumerate(cols):
                    if i < len(st.session_state.m_images):
                        col.image(st.session_state.m_images[i], caption=f"M {i+1}")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.stop()
        
        # Initialize predictor
        model_dir = "models"
        predictor = MPredictor(model_dir)
        
        # Analyze each M
        if st.button("Analyze Thinking Patterns"):
            with st.spinner("Analyzing thinking patterns..."):
                st.session_state.results = []
                confidence_sums = {
                    'Cumulative': 0,
                    'Investigative': 0,
                    'Comprehensive': 0,
                    'Analytical': 0
                }
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, m_image in enumerate(st.session_state.m_images, 1):
                    status_text.text(f"Analyzing M {i} of {len(st.session_state.m_images)}...")
                    progress_bar.progress(i / len(st.session_state.m_images))
                    
                    prediction, probs = predictor.predict(m_image)
                    
                    if prediction is None:
                        continue
                    
                    # Store probabilities for each pattern
                    class_probs = dict(zip(predictor.le.classes_, probs))
                    
                    # Add to confidence sums
                    for pattern in confidence_sums:
                        confidence_sums[pattern] += class_probs.get(pattern, 0)
                    
                    st.session_state.results.append({
                        'M_number': i,
                        'Prediction': prediction,
                        'Confidence': max(probs),
                        'Probabilities': class_probs
                    })
                
                # Calculate weighted confidence scores
                total_confidence = sum(confidence_sums.values())
                if total_confidence > 0:
                    st.session_state.normalized_scores = {
                        k: (v/total_confidence)*100 for k, v in confidence_sums.items()
                    }
                else:
                    st.session_state.normalized_scores = {k: 0 for k in confidence_sums.keys()}
                
                progress_bar.empty()
                status_text.empty()
                st.success("Analysis complete!")
        
        # Display results if available
        if st.session_state.results:
            st.subheader("Analysis Results")
            
            # Show individual M results in an expander
            with st.expander("View Detailed Results for Each 'M'"):
                for result in st.session_state.results:
                    st.markdown(f"**M {result['M_number']}**")
                    st.markdown(f"- **Predicted Pattern**: {result['Prediction']}")
                    st.markdown(f"- **Confidence**: {result['Confidence']*100:.1f}%")
                    
                    # Create a DataFrame for the probabilities for better display
                    prob_df = pd.DataFrame.from_dict(
                        result['Probabilities'], 
                        orient='index', 
                        columns=['Probability']
                    )
                    prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x*100:.1f}%")
                    st.dataframe(prob_df, use_container_width=True)
                    st.markdown("---")
            
            # Show aggregated results
            st.subheader("Thinking Pattern Distribution")
            
            # Create two columns for the chart and metrics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Plot the distribution
                fig, ax = plt.subplots(figsize=(8, 4))
                patterns = list(st.session_state.normalized_scores.keys())
                percentages = list(st.session_state.normalized_scores.values())
                bars = ax.bar(patterns, percentages, 
                             color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                
                # Add value labels on top of each bar
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%',
                            ha='center', va='bottom')
                
                ax.set_title('Confidence-Weighted Thinking Pattern Distribution')
                ax.set_ylabel('Percentage Score')
                ax.set_ylim(0, 100)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)
            
            with col2:
                # Show dominant pattern
                dominant_pattern = max(
                    st.session_state.normalized_scores.items(), 
                    key=lambda x: x[1]
                )
                st.metric(
                    label="Dominant Thinking Pattern",
                    value=dominant_pattern[0],
                    delta=f"{dominant_pattern[1]:.1f}%"
                )
                
                # Show all scores as metrics
                st.markdown("**All Pattern Scores:**")
                for pattern, score in st.session_state.normalized_scores.items():
                    st.metric(label=pattern, value=f"{score:.1f}%")
            
            # Interpretation of results
            st.subheader("Interpretation")
            st.markdown("""
            - **Cumulative (25-50%)**: Indicates a methodical, step-by-step thinking style.
            - **Investigative (25-50%)**: Suggests a questioning, exploratory approach.
            - **Comprehensive (25-50%)**: Shows big-picture, holistic thinking.
            - **Analytical (25-50%)**: Reflects logical, detail-oriented processing.
            """)
            
            # Recommendation based on dominant pattern
            dominant = dominant_pattern[0]
            if dominant == "Cumulative":
                st.info("""
                **Recommendation**: This thinking style benefits from structured environments 
                and clear sequences. Provide step-by-step instructions and timelines.
                """)
            elif dominant == "Investigative":
                st.info("""
                **Recommendation**: This exploratory style thrives on open-ended questions 
                and research opportunities. Encourage questioning and investigation.
                """)
            elif dominant == "Comprehensive":
                st.info("""
                **Recommendation**: This big-picture thinking benefits from seeing how 
                parts relate to the whole. Provide overviews and context.
                """)
            elif dominant == "Analytical":
                st.info("""
                **Recommendation**: This detail-oriented style excels with data and 
                logical arguments. Provide evidence and precise information.
                """)