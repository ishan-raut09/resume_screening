import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Global variable to cache the SentenceTransformer model and avoid reloading on every function call
_transformer_model = None

def get_sentence_transformer():
    """Lazy load the sentence transformer model and cache it."""
    global _transformer_model
    if _transformer_model is None:
        print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        # Use sentence-transformers to load the model
        _transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _transformer_model

def calculate_ats_score(resume_skills, required_skills):
    """
    ATS Score = (Number of Matched Skills / Required Skills) * 100
    """
    if not required_skills:
        return 0.0
    matched_skills = set(resume_skills).intersection(set(required_skills))
    score = (len(matched_skills) / len(required_skills)) * 100.0
    return round(score, 1)

def calculate_semantic_similarity(resume_text, jd_text):
    """
    Compute semantic similarity between resume and job description using Sentence Transformers.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    
    model = get_sentence_transformer()
    embeddings = model.encode([resume_text, jd_text])
    
    # Calculate cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Clip similarity to [0, 1] range (negative similarity isn't meaningful for this UI metric)
    similarity_percentage = max(0.0, float(similarity)) * 100.0
    return round(similarity_percentage, 1)

def calculate_final_score(ats_score, similarity_score):
    """
    Final Score = 70% ATS Score + 30% Similarity Score
    """
    final = 0.7 * ats_score + 0.3 * similarity_score
    return round(final, 1)

def analyze_resume(resume_text, resume_skills, jd_text, required_skills):
    """
    Perform a complete analysis of a resume text against a job description.
    """
    matched_skills = sorted(list(set(resume_skills).intersection(set(required_skills))))
    missing_skills = sorted(list(set(required_skills) - set(resume_skills)))
    
    ats_score = calculate_ats_score(resume_skills, required_skills)
    similarity_score = calculate_semantic_similarity(resume_text, jd_text)
    final_score = calculate_final_score(ats_score, similarity_score)
    
    skill_match_percentage = round((len(matched_skills) / len(required_skills)) * 100, 1) if required_skills else 0.0
    
    return {
        "ats_score": ats_score,
        "similarity_score": similarity_score,
        "final_score": final_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_match_percentage": skill_match_percentage
    }
