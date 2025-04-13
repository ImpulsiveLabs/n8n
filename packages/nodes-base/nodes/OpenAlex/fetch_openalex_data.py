import json
import os
import sys
import re
from tqdm import tqdm
from pyalex import Works

def clean_text(text):
    """Ensure text is a string and clean it."""
    if text is None or not isinstance(text, str):
        return ""
    text = re.sub(r'\\u003[CE]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

import sys

def is_relevant(work, relevant_terms, exclude_terms):
    """
    Determines if a work is relevant based on the provided relevant and exclude terms.

    Args:
        work (dict): The work to evaluate (contains title, topics, keywords).
        relevant_terms (dict): A dictionary where keys are categories, and values are lists of relevant terms.
        exclude_terms (set): A set of terms to exclude from relevance.

    Returns:
        bool: True if the work is relevant, False otherwise.
    """
    # Clean the title, topics, and keywords
    title = clean_text(work.get('title', '')).lower()
    topics = {clean_text(t.get('display_name', '')).lower() for t in work.get('topics', [])}
    keywords = {clean_text(k.get('display_name', '')).lower() for k in work.get('keywords', [])}
    all_text = {title} | topics | keywords


    # Check if the work matches any of the relevant categories
    matching_categories = 0
    for category, terms in relevant_terms.items():
        if any(term in text for text in all_text for term in terms):
            matching_categories += 1

    # Check exclusion terms
    is_excluded = any(term in text for text in all_text for term in exclude_terms)

    # Require at least two relevant categories to be matched
    is_relevant = matching_categories >= 2


    # Return True if relevant terms are matched (at least two of the categories) and not excluded
    return is_relevant and not is_excluded

def fetch_openalex_data(query, relevant_terms, exclude_terms):
    filters = {"title_and_abstract.search": query, "is_oa": True}
    all_works = []
    paginator = Works().filter(**filters).sort(relevance_score="desc").paginate(per_page=100)
    for page in tqdm(paginator, desc="Fetching pages"):
        all_works.extend(page)
    relevant_works = [
        work for work in tqdm(all_works, desc="Filtering")
        if is_relevant(work, relevant_terms, exclude_terms)
    ]
    return relevant_works

if __name__ == "__main__":
    QUERY = os.getenv("ARG1", "")
    RELEVANT_TERMS = json.loads(os.getenv("ARG2", "{}"))
    EXCLUDE_TERMS = json.loads(os.getenv("ARG3", "{}"))

    data = fetch_openalex_data(QUERY, RELEVANT_TERMS, EXCLUDE_TERMS)
    json_output = json.dumps(data, ensure_ascii=False, default=str)
    print(json_output if data else "[]", flush=True)
    sys.stdout.flush()
