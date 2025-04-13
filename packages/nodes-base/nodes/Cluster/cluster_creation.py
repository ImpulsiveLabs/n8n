import os
import sys
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def clean_text(text):
    """
    Cleans input text by:
    - Removing specific unicode sequences.
    - Removing HTML tags.
    - Replacing multiple spaces with a single space.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\\u003[CE]', '', text)  # Removes specific unicode sequences
    text = re.sub(r'<[^>]+>', '', text)    # Removes HTML tags
    text = re.sub(r'\s+', ' ', text).strip()  # Replaces multiple spaces with a single space
    return text
# Construct text for embedding
def construct_text_for_embedding(work):
    title = clean_text(work.get('title', ''))
    topics = " ".join(clean_text(t.get('display_name', '')) for t in work.get('topics', []))
    text = " ".join([title, topics]).strip()
    return text if text else "management decision visualization"

# Generate embeddings and clusters
def generate_embeddings_and_clusters(texts, MODEL):
    model = SentenceTransformer(MODEL)
    embeddings = model.encode(texts, show_progress_bar=True)
    distance_matrix = cosine_distances(embeddings).astype(np.float64)

    # HDBSCAN with relaxed parameters
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=3, min_samples=2, cluster_selection_method='eom')
    clusters = clusterer.fit_predict(distance_matrix)

    # Fallback to K-means if too few clusters
    unique_clusters = set(clusters) - {-1}
    if len(unique_clusters) < 3 and len(texts) >= 9:  # Min 3 clusters, need 9 works for 3 clusters of 3
        # print("HDBSCAN produced too few clusters, falling back to K-means...")
        n_clusters = min(7, max(3, len(texts) // 3))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

    # Cap at 7 clusters
    cluster_map = {}
    for idx, cid in enumerate(clusters):
        if cid != -1:
            cluster_map.setdefault(cid, []).append(idx)
    if len(cluster_map) > 7:
        sorted_clusters = sorted(cluster_map.items(), key=lambda x: len(x[1]))
        for cid, indices in sorted_clusters[:len(cluster_map) - 7]:
            for idx in indices:
                clusters[idx] = -1

    return embeddings, clusters

# Extract key phrases
def get_key_phrases(texts):
    if not texts or len(texts) < 2:
        return []
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words='english', max_features=3, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out().tolist()

# Generate main idea and cluster name
def get_main_idea_and_name(cluster_items):
    titles = [clean_text(item.get('title', '')) for item in cluster_items]
    topics = [clean_text(t.get('display_name', '')) for item in cluster_items for t in item.get('topics', [])]
    combined_text = titles + topics
    key_phrases = get_key_phrases(combined_text)
    # print(f"Cluster phrases: {key_phrases}")
    if not key_phrases:
        return "No distinct theme identified.", "No Theme"
    if len(key_phrases) == 1:
        return f"{key_phrases[0]} is the core focus.", key_phrases[0]
    if len(key_phrases) == 2:
        return f"{key_phrases[0]} and {key_phrases[1]} define this group.", key_phrases[0]
    return (f"{key_phrases[0]} drives advancements in {key_phrases[1]} and {key_phrases[2]}.", key_phrases[0])

# Calculate intra-cluster similarity
def cosine_similarity(vecA, vecB):
    dot_product = np.dot(vecA, vecB)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    return dot_product / (normA * normB) if normA and normB else 0

def calculate_intra_cluster_similarity(embeddings, cluster_indices):
    if len(cluster_indices) < 2:
        return 1.0
    total_sim = 0
    count = 0
    for i in range(len(cluster_indices) - 1):
        for j in range(i + 1, len(cluster_indices)):
            total_sim += cosine_similarity(embeddings[cluster_indices[i]], embeddings[cluster_indices[j]])
            count += 1
    return total_sim / count

def create_network_data(cluster_map, relevant_works, embeddings):
    """
    Generates network data for visualization with nodes and edges,
    including intra-cluster similarity and main ideas.
    """
    nodes = []
    clusters_info = []

    for cluster_id, cluster_items in cluster_map.items():
        cluster_titles = [clean_text(item['work'].get('title', '')) for item in cluster_items]
        cluster_indices = [item['index'] for item in cluster_items]
        intra_sim = calculate_intra_cluster_similarity(embeddings, cluster_indices)
        main_idea, cluster_name = get_main_idea_and_name([item['work'] for item in cluster_items])

        clusters_info.append({
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'main_idea': main_idea,
            'intra_cluster_similarity': intra_sim,
            'titles': cluster_titles
        })

        for item in cluster_items:
            nodes.append({
                'id': clean_text(item['work'].get('title', '')),
                'cluster': cluster_name
            })

    return {
        'nodes': nodes,
        'clusters_info': clusters_info
    }
def load_input_file(input_file):
    """
    Loads the input file and ensures it is parsed correctly.
    If the file contains stringified JSON, it will parse it again.
    """
    try:
        with open(input_file, "r") as f:
            data = json.load(f)

        # Check if the data is a string, if so parse it again
        if isinstance(data, str):
            data = json.loads(data)

        if not isinstance(data, list):
            print("❌ ERROR: Expected 'all_works' to be a list.", file=sys.stderr)
            return None

        return data
    except json.JSONDecodeError:
        print("❌ ERROR: Failed to decode JSON from the input file.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"❌ ERROR: An exception occurred while loading the input file: {str(e)}", file=sys.stderr)
        return None

# def create_network_data(cluster_map, relevant_works, embeddings):
#     """
#     Generates network data for visualization with nodes and edges.
#     """
#     nodes = []
#     edges = []

#     for cluster_id, cluster_items in cluster_map.items():
#         # Convert numpy.int32 to native int for JSON serialization
#         cluster_id = int(cluster_id)  # Ensure cluster_id is a regular int
#         cluster_name = f"Cluster {cluster_id}"  # Placeholder for cluster name or main idea
#         for item in cluster_items:
#             nodes.append({
#                 'id': item['index'],
#                 'label': clean_text(item['work'].get('title', '')),
#                 'cluster': cluster_name
#             })
#             for other_item in cluster_items:
#                 if item['index'] != other_item['index']:
#                     edges.append({
#                         'source': item['index'],
#                         'target': other_item['index'],
#                         'weight': np.dot(embeddings[item['index']], embeddings[other_item['index']])
#                     })

#     return {'nodes': nodes, 'edges': edges}

def cluster_works(input_file, MODEL):
    """
    Main function to load input file, process the data, and generate clusters.
    """
    try:
        # Load the input file
        all_works = load_input_file(input_file)
        if not all_works:
            return {}

        # Construct text for embeddings from the 'title' and 'topics' fields
        texts = [construct_text_for_embedding(work) for work in all_works]

        # Generate embeddings and clusters
        embeddings, clusters = generate_embeddings_and_clusters(texts, MODEL)

        # Organize the works into clusters, ignoring noise points (cluster_id == -1)
        cluster_map = {}
        for idx, cluster_id in enumerate(clusters):  # <- Fixed indentation
            if cluster_id == -1:
                continue
            if cluster_id not in cluster_map:
                cluster_map[cluster_id] = []
            cluster_map[cluster_id].append({'index': idx, 'work': all_works[idx]})  # <- Fixed `relevant_works` reference

        # print(f"Number of clusters: {len(cluster_map)}, Noise points: {sum(1 for c in clusters if c == -1)}")
        for cluster_id, cluster_items in cluster_map.items():
            cluster_titles = [clean_text(item['work'].get('title', '')) for item in cluster_items]
            cluster_indices = [item['index'] for item in cluster_items]
            intra_sim = calculate_intra_cluster_similarity(embeddings, cluster_indices)
            main_idea, cluster_name = get_main_idea_and_name([item['work'] for item in cluster_items])

            # print(f"\nCluster '{cluster_name}':")
            # print(f"Titles: {', '.join(cluster_titles)}")
            # print(f"Main Idea: {main_idea}")
            # print(f"Intra-cluster Similarity: {intra_sim:.4f}")

        network_data = create_network_data(cluster_map, all_works, embeddings)
        return network_data

    except Exception as e:
        print(f"❌ ERROR: An exception occurred during clustering: {str(e)}", file=sys.stderr)
        return {}

if __name__ == "__main__":
    """
    Entry point for the script. It expects an input file as a command-line argument.
    """
    if len(sys.argv) < 2:
        print("❌ ERROR: No input file provided", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    MODEL = os.getenv("ARG1", "")
    result = cluster_works(input_file, MODEL)

    if result:
        # Ensure cluster IDs are properly converted and print the result
        result = json.dumps(result, ensure_ascii=False, indent=2, default=str)  # Converts numpy.int32 to string
        print(result, flush=True)
    else:
        print("❌ ERROR: Clustering failed or no valid data.", file=sys.stderr)

    sys.stdout.flush()
