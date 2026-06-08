import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import shutil

from utils.parser import extract_text, extract_name, extract_email, extract_phone
from utils.skills import extract_skills
from utils.scorer import analyze_resume

# Page configuration
st.set_page_config(
    page_title="AI-Powered Resume Screener",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for vibrant premium look
st.markdown("""
<style>
    .main {
        background-color: #f9fafd;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #eef2f6;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 5px;
    }
    .matched-skill {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }
    .missing-skill {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Temporary directory for uploaded resumes
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper function to load model
@st.cache_resource
def load_classifier_pipeline():
    pipeline_path = os.path.join("model", "resume_pipeline.pkl")
    if os.path.exists(pipeline_path):
        try:
            with open(pipeline_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

# Helper to load category distribution from CSV
@st.cache_data
def load_category_distribution():
    # Try different paths
    csv_paths = [
        os.path.join("dataset", "Resume", "Resume.csv"),
        os.path.join("dataset", "resume_dataset.csv")
    ]
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if "Category" in df.columns:
                    return df["Category"].value_counts()
                elif "category" in df.columns:
                    return df["category"].value_counts()
            except Exception as e:
                print(f"Error reading CSV for chart: {e}")
    return None

# Sidebar Content
st.sidebar.title("🛠️ System Status")

# Load model pipeline
pipeline = load_classifier_pipeline()
model_loaded = pipeline is not None

if model_loaded:
    st.sidebar.success("✅ SVM Classifier Loaded")
    
    # Read test accuracy if available
    accuracy_file = os.path.join("model", "classification_report.txt")
    if os.path.exists(accuracy_file):
        try:
            with open(accuracy_file, "r") as f:
                first_line = f.readline().strip()
                if "Accuracy" in first_line:
                    st.sidebar.metric("Model Test Accuracy", first_line.split(":")[-1].strip())
        except Exception:
            pass
            
    # Confusion matrix expander
    cm_path = os.path.join("model", "confusion_matrix.png")
    if os.path.exists(cm_path):
        with st.sidebar.expander("📊 View Model Confusion Matrix"):
            st.image(cm_path, caption="Confusion Matrix (Linear SVM)", use_container_width=True)
else:
    st.sidebar.warning("⚠️ Model Pipeline Not Found")
    st.sidebar.info("Please run the training script first to train and generate the classifier pipeline:\n`python train_classifier.py`")
    
    # Add a train button directly in Streamlit
    if st.sidebar.button("🚀 Train Model Now"):
        with st.spinner("Training model on Resume Dataset... This may take a moment."):
            try:
                from train_classifier import train
                train()
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success("Model trained successfully! Please reload the page.")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

st.sidebar.divider()
st.sidebar.markdown("""
**AI Resume Screener & Candidate Ranker**
- **ATS Score**: Matches resume skills against JD (70% weight)
- **Semantic Score**: Contextual match via Sentence Transformers (30% weight)
- **Category Match**: Predicts candidate stream using SVM
""")

# Title Banner
st.title("💼 AI-Powered Resume Screening & Candidate Ranking")
st.markdown("Rank candidates contextually using natural language similarity, extract matching/missing skills, and predict professional domain streams.")

# Show dataset insights
category_counts = load_category_distribution()
if category_counts is not None:
    with st.expander("📊 Dataset Domain Insights (Training Distribution)", expanded=False):
        st.write(f"The classifier is trained on **{category_counts.sum()}** resumes across **{len(category_counts)}** categories.")
        st.bar_chart(category_counts)

# Main Form Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Enter Job Description")
    
    default_jd = """Python Developer

Required Skills:
Python
SQL
Machine Learning
AWS
Docker
Git
Kubernetes

Description:
We are looking for a Senior Python Developer to build and deploy robust machine learning applications. You will work with SQL databases, package microservices using Docker and Kubernetes, and manage cloud infrastructure on AWS. Strong experience in software engineering best practices and Git version control is essential.
"""
    jd_input = st.text_area(
        "Paste the Job Description here:",
        value=default_jd,
        height=320,
        placeholder="Enter details of required skills and qualifications..."
    )
    
    # Extract JD required skills
    jd_skills = extract_skills(jd_input)
    
    st.markdown("**Detected Required Skills:**")
    if jd_skills:
        skills_html = "".join([f'<span class="matched-skill">{s}</span>' for s in jd_skills])
        st.markdown(skills_html, unsafe_allow_html=True)
    else:
        st.warning("No predefined skills detected in the job description. Add skills like 'Python', 'AWS', or 'SQL' to analyze matching metrics.")

with col2:
    st.subheader("2. Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload Candidate Resumes (PDF or DOCX):",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="You can upload multiple files at once."
    )
    
    if uploaded_files:
        st.success(f"📂 {len(uploaded_files)} files selected.")
        # Listing files selected
        file_details = []
        for file in uploaded_files:
            file_details.append({"Filename": file.name, "Size (KB)": round(file.size / 1024, 1)})
        st.dataframe(pd.DataFrame(file_details), use_container_width=True, hide_index=True)
    else:
        st.info("Please upload one or more resumes to begin candidate ranking.")

# Trigger Analysis
st.markdown("---")
analyze_button = st.button("🔍 Analyze & Rank Candidates", type="primary", use_container_width=True)

if analyze_button:
    if not model_loaded:
        st.error("Cannot perform analysis: The classification model pipeline is not trained or loaded. Please train the model in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume to analyze.")
    elif not jd_skills:
        st.warning("Job Description has 0 matching skills detected. Please add some required skills in the description.")
    else:
        st.subheader("🏆 Candidate Ranking Dashboard")
        
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        candidates_data = []
        
        # Save files and process
        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Processing candidate {idx + 1}/{len(uploaded_files)}: {file.name}")
            
            # Save file locally
            temp_path = os.path.join(UPLOAD_DIR, file.name)
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
                
            try:
                # Extract text
                resume_text = extract_text(temp_path)
                
                if not resume_text.strip():
                    st.error(f"Failed to extract text from {file.name}. File might be empty or scanned image.")
                    continue
                
                # Extract Contact Info
                c_name = extract_name(resume_text)
                if c_name == "Not Found": 
                    c_name = file.name
                c_email = extract_email(resume_text)
                c_phone = extract_phone(resume_text)
                
                # Predict Category
                predicted_category = pipeline.predict([resume_text])[0]
                
                # Extract Skills
                resume_skills = extract_skills(resume_text)
                
                # Analyze Scorer
                analysis = analyze_resume(resume_text, resume_skills, jd_input, jd_skills)
                
                candidates_data.append({
                    "filename": file.name,
                    "name": c_name,
                    "email": c_email,
                    "phone": c_phone,
                    "predicted_category": predicted_category,
                    "ats_score": analysis["ats_score"],
                    "similarity_score": analysis["similarity_score"],
                    "final_score": analysis["final_score"],
                    "matched_skills": analysis["matched_skills"],
                    "missing_skills": analysis["missing_skills"],
                    "skill_match_percentage": analysis["skill_match_percentage"],
                    "text": resume_text
                })
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
                
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        status_text.empty()
        progress_bar.empty()
        
        if candidates_data:
            # Sort candidates by final score
            candidates_df = pd.DataFrame(candidates_data)
            candidates_df = candidates_df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
            candidates_df["rank"] = candidates_df.index + 1
            
            # Show ranking table
            display_df = candidates_df[["rank", "name", "email", "phone", "predicted_category", "ats_score", "similarity_score", "final_score"]].copy()
            display_df.columns = ["Rank", "Candidate Name", "Email", "Phone", "Predicted Category", "ATS Score (%)", "Semantic Similarity (%)", "Final Score (%)"]
            
            st.dataframe(
                display_df.style.highlight_max(subset=["Final Score (%)"], color="#dcfce7"),
                use_container_width=True,
                hide_index=True
            )
            
            # Analytics Dashboard / Bar Chart
            st.markdown("### 📊 ATS Score Comparison")
            chart_data = candidates_df.set_index("name")[["ats_score", "similarity_score", "final_score"]]
            chart_data.columns = ["ATS Score", "Semantic Similarity", "Final Score"]
            st.bar_chart(chart_data)
            
            # Clear upload directory files after loading in memory
            for file in os.listdir(UPLOAD_DIR):
                try:
                    os.remove(os.path.join(UPLOAD_DIR, file))
                except Exception:
                    pass
            
            # Detailed Drill Down
            st.markdown("### 🔍 Detailed Candidate Breakdown")
            for index, row in candidates_df.iterrows():
                expander_title = f"Rank #{row['rank']}: {row['name']} — Final Score: {row['final_score']}%"
                
                with st.expander(expander_title):
                    c_col1, c_col2 = st.columns([1, 1])
                    
                    with c_col1:
                        st.markdown(f"**Predicted Stream (Category):** `{row['predicted_category']}`")
                        
                        # Skill match % and progress bar
                        match_pct = row['skill_match_percentage'] / 100.0
                        st.markdown(f"**Skill Match Percentage:** `{row['skill_match_percentage']}%`")
                        st.progress(match_pct)
                        
                        # Displaying metrics side by side
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.metric("ATS Score", f"{row['ats_score']}%")
                        with m_col2:
                            st.metric("Semantic Similarity", f"{row['similarity_score']}%")
                            
                    with c_col2:
                        st.markdown("**Skill Gap Analysis:**")
                        
                        st.markdown("✅ **Matched Skills:**")
                        if row['matched_skills']:
                            matched_html = "".join([f'<span class="matched-skill">{s}</span>' for s in row['matched_skills']])
                            st.markdown(matched_html, unsafe_allow_html=True)
                        else:
                            st.write("None")
                            
                        st.markdown("❌ **Missing Skills:**")
                        if row['missing_skills']:
                            missing_html = "".join([f'<span class="missing-skill">{s}</span>' for s in row['missing_skills']])
                            st.markdown(missing_html, unsafe_allow_html=True)
                        else:
                            st.success("No missing skills! Perfect match.")
                    
                    st.divider()
                    st.markdown("**Extracted Resume Text (Snippet):**")
                    st.text_area(
                        "Parsed Resume Content",
                        value=row['text'][:1500] + ("..." if len(row['text']) > 1500 else ""),
                        height=200,
                        key=f"text_area_{index}",
                        disabled=True
                    )
        else:
            st.warning("No resumes were successfully processed. Please verify your uploaded files.")
