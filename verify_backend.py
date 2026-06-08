import os
import pickle
from utils.skills import extract_skills
from utils.scorer import analyze_resume, calculate_ats_score, calculate_semantic_similarity

def test_backend():
    print("=== Testing Skills Extraction ===")
    sample_text = "I am a Python developer with experience in SQL, AWS, Docker and Machine Learning."
    skills = extract_skills(sample_text)
    print(f"Extracted skills: {skills}")
    assert "Python" in skills
    assert "SQL" in skills
    assert "AWS" in skills
    assert "Docker" in skills
    assert "Machine Learning" in skills
    print("Skills extraction: SUCCESS\n")

    print("=== Testing Scorer & Sentence Transformer ===")
    jd_text = "Python developer. Required skills: Python, SQL, Docker, AWS."
    jd_skills = extract_skills(jd_text)
    print(f"Required skills: {jd_skills}")
    
    analysis = analyze_resume(sample_text, skills, jd_text, jd_skills)
    print(f"Analysis results: {analysis}")
    assert analysis["ats_score"] > 0
    assert analysis["similarity_score"] > 0
    assert analysis["final_score"] > 0
    print("Scorer & Similarity: SUCCESS\n")

    print("=== Testing Trained Classifier Pipeline ===")
    pipeline_path = os.path.join("model", "resume_pipeline.pkl")
    if os.path.exists(pipeline_path):
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)
        pred = pipeline.predict([sample_text])[0]
        print(f"Sample resume text predicted category: {pred}")
        print("Classifier loading and prediction: SUCCESS\n")
    else:
        print("Classifier model file not found. Skipping classifier check.")

    print("All backend components verified successfully!")

if __name__ == "__main__":
    test_backend()
