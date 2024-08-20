from text_processing import *

# Test the compute_cosine_similarity function with dummy data

# Dummy data for testing
text1 = "The quick brown fox jumps over the lazy dog."
text2 = "A fast brown fox leaps over a lazy dog."
text3 = "The quick brown fox is very quick and brown."

# Compute cosine similarity between text1 and text2
similarity_1_2 = compute_cosine_similarity(text1, text2)
print(f"Cosine Similarity between text1 and text2: {similarity_1_2}")

# Compute cosine similarity between text1 and text3
similarity_1_3 = compute_cosine_similarity(text1, text3)
print(f"Cosine Similarity between text1 and text3: {similarity_1_3}")

# Compute cosine similarity between text2 and text3
similarity_2_3 = compute_cosine_similarity(text2, text3)
print(f"Cosine Similarity between text2 and text3: {similarity_2_3}")