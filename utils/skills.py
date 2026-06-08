import re

PREDEFINED_SKILLS = [
    "Python", "Java", "C++", "JavaScript", "React", "Node.js", "SQL",
    "MongoDB", "PostgreSQL", "AWS", "Azure", "Docker", "Kubernetes",
    "Git", "Linux", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch"
]

def clean_text(text):
    if not text:
        return ""
    # Lowercase and replace newlines/multiple spaces
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_skills(text):
    """
    Extract skills from text based on a predefined dictionary.
    Uses regex to match skills as whole words or phrases, preventing false matches.
    """
    cleaned = clean_text(text)
    extracted = []
    
    for skill in PREDEFINED_SKILLS:
        skill_lower = skill.lower()
        escaped = re.escape(skill_lower)
        
        # Handle custom word boundary regex for special characters
        if skill_lower == "c++":
            # Match 'c++' with a word boundary at the start, but not at the end because '+' is non-alphanumeric
            pattern = r'\bc\+\+'
        elif skill_lower == "node.js":
            pattern = r'\bnode\.js\b'
        elif skill_lower == "deep learning":
            pattern = r'\bdeep\s+learning\b'
        elif skill_lower == "machine learning":
            pattern = r'\bmachine\s+learning\b'
        else:
            pattern = r'\b' + escaped + r'\b'
            
        if re.search(pattern, cleaned):
            extracted.append(skill)
            
    return extracted
