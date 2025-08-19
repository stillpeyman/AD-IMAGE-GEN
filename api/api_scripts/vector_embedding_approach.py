import json
import os
from math import sqrt

from dotenv import load_dotenv
from openai import OpenAI
from .gpt_pipeline import analyze_product_image
from .rapid_api_trending_hashtags import fetch_trending_hashtags


# Load environment variables so the API key(s) are available from .env
load_dotenv()
api_key = os.getenv("MY_OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# Run product image analysis to obtain semantic keywords about the product.
# We compute an absolute path to ensure it works regardless of current working directory.
image_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "test_images", "nike-unsplash.jpg"
    )
)
result = analyze_product_image(image_path)
suggested_keywords = result.advertising_keywords  # list[str]: semantic descriptors from analysis

# Fetch trending hashtags (recent, WORLD region) via Rapid API.
hashtags_data = fetch_trending_hashtags("7", "WORLD")
trending_hashtags = [item["hashtag"] for item in hashtags_data]

# Embed the keywords. Each entry in .data has an `.embedding` list[float] of length 1536.
keyword_embeddings = client.embeddings.create(
    input=suggested_keywords,
    model="text-embedding-3-small"
).data

# Embed the hashtags, same model to keep the vectors in the same space.
hashtag_embeddings = client.embeddings.create(
    input=trending_hashtags,
    model="text-embedding-3-small"
).data

def _to_vectors(emb_list):
    """Extract raw float vectors from the API embedding objects.

    The API returns a list of objects, each with an `.embedding` attribute holding
    the actual vector (list[float]). This helper converts
    [EmbeddingObject, ...] -> [[float, ...], ...] while preserving order.
    """
    return [e.embedding for e in emb_list]

def _cosine(u, v):
    """Compute cosine similarity between two vectors u and v.

    - dot product = Σ(u_i * v_i): elementwise multiply and sum (via zip to pair entries)
    - nu, nv: magnitudes (Euclidean norms) of u and v, respectively
      We name them `nu` and `nv` per math shorthand (norm of u/v). You could use
      `magnitude_u`/`magnitude_v` for extra clarity; behavior is identical.
    - Return dot / (nu * nv) when both magnitudes are non-zero. If either vector
      had zero length (unlikely for model embeddings), return 0.0 to avoid division by zero.
    """
    dot = sum(a * b for a, b in zip(u, v))
    nu = sqrt(sum(a * a for a in u))  # |u|
    nv = sqrt(sum(b * b for b in v))  # |v|
    return dot / (nu * nv) if nu and nv else 0.0

def _top_k(vec, labels, vectors, k=5):
    """Return the top-k most similar labeled vectors to a single query vector.

    Parameters:
    - vec: list[float] — the single query vector (e.g., one keyword embedding)
    - labels: list[str] — the names corresponding to `vectors` (e.g., hashtag texts)
    - vectors: list[list[float]] — embeddings aligned 1:1 with `labels`
    - k: how many best matches to keep

    We compute cosine similarity between `vec` and every vector `v` in `vectors`,
    pair each score with its label, sort descending by similarity, then slice the top k.
    """
    similarities = [(label, _cosine(vec, v)) for label, v in zip(labels, vectors)]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

kw_vecs = _to_vectors(keyword_embeddings)  # [[float,...], ...] for keywords
ht_vecs = _to_vectors(hashtag_embeddings)  # [[float,...], ...] for hashtags

print("\n=== Embedding summary ===")  # human-friendly section header
print(f"Keywords: {len(suggested_keywords)}")
print(f"Hashtags: {len(trending_hashtags)}")
print(
    f"Embedding dims: {len(kw_vecs[0]) if kw_vecs else 0}"
)
# "Embedding dims" is the vector length produced by the model (1536 for text-embedding-3-small).
# All vectors from the same model have the same dimensionality, so we inspect the first.

print("\nKeywords:")  # echo keywords to correlate with results below
print(", ".join(suggested_keywords) or "(none)")

print("\nTrending hashtags:")  # echo hashtags for transparency
print(", ".join(trending_hashtags) or "(none)")

print("\n=== Top matches (first 5 keywords) ===")
for i, kw in enumerate(suggested_keywords[:5]):
    # i is the positional index of the current keyword; kw_vecs[i] selects its embedding.
    # We compare this one keyword vector against all hashtag vectors, then keep the best 5.
    matches = _top_k(kw_vecs[i], trending_hashtags, ht_vecs, k=5)
    line = ", ".join(f"{tag} ({score:.2f})" for tag, score in matches)
    # {i+1:>2} right-aligns the 1-based index in a width of 2 characters for neat columns.
    print(f"{i+1:>2}. {kw}: {line}")

# Save full results for inspection
results_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "embedding_results.json")
)
full_results = {
    "embedding_dims": len(kw_vecs[0]) if kw_vecs else 0,
    "keywords": suggested_keywords,
    "hashtags": trending_hashtags,
    "top_matches": {
        kw: _top_k(kw_vecs[i], trending_hashtags, ht_vecs, k=10)
        for i, kw in enumerate(suggested_keywords)
    },
}
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(full_results, f, indent=2, ensure_ascii=False)
print(f"\nSaved full similarity results to: {results_path}")


