# AI-Powered Resume Screening and Candidate Ranking System 💼🤖

A working MVP of an AI-Powered Resume Screening and Candidate Ranking System. This tool enables recruiters to upload multiple resumes (PDF/DOCX), paste a job description, extract matching skills, run a machine learning classifier to predict the candidate's category, perform semantic similarity matching using Sentence Transformers, and rank candidates based on a weighted score.

## Features

1. **Resume Category Prediction**: Multi-class categorization using a Linear SVM model trained on the Kaggle Resume Dataset (90%+ category classification accuracy).
2. **Resume Upload**: Extends support to PDF and DOCX formats using `pdfplumber`, `PyPDF2`, and `python-docx` for reliable parsing.
3. **Skill Extraction**: Matches required skills from the job description against the resume based on a predefined dictionary of 19 common technical skills.
4. **ATS Score**: Calculates the percentage match of required skills in the resume:
   $$\text{ATS Score} = \left(\frac{\text{Matched Skills}}{\text{Required Skills}}\right) \times 100$$
5. **Semantic Similarity**: Captures context using `SentenceTransformer('all-MiniLM-L6-v2')` and calculates cosine similarity between resume text and job description.
6. **Candidate Ranking**: Ranks candidates based on a weighted score:
   $$\text{Final Score} = 70\% \times \text{ATS Score} + 30\% \times \text{Similarity Score}$$
7. **Skill Gap Analysis**: Visualizes matched and missing skills with intuitive badges and a progress bar showing the skill match percentage.

---

## Project Structure

```
resume_screening/
├── app.py                     # Streamlit frontend application
├── train_classifier.py        # Model training and evaluation script
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── dataset/
│   └── Resume/
│       └── Resume.csv         # Kaggle Resume Dataset
├── model/
│   ├── resume_pipeline.pkl    # Trained classification pipeline (TF-IDF + LinearSVC)
│   ├── confusion_matrix.png   # Model evaluation: Confusion Matrix plot
│   └── classification_report.txt # Model evaluation: Metrics report
└── utils/
    ├── parser.py              # Resume PDF/DOCX text extraction
    ├── skills.py              # Skill dictionary & matching logic
    └── scorer.py              # ATS & semantic similarity score calculation
```

---

## Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Classifier**:
   Ensure `dataset/Resume/Resume.csv` is present. Run the training script to train the model, save the pipeline, and output the confusion matrix + classification report:
   ```bash
   python train_classifier.py
   ```

3. **Launch the Streamlit App**:
   ```bash
   python -m streamlit run app.py
   ```

---

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python (Pandas, Numpy)
- **Machine Learning**: Scikit-learn, Sentence Transformers (`all-MiniLM-L6-v2`)
- **Parsers**: `pdfplumber`, `PyPDF2`, `python-docx`
- **Visualization**: Matplotlib, Seaborn
