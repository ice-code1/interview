from sentence_transformers import SentenceTransformer


# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode a sample description to get the vector
sample_description = "white metal lantern"
sample_vector = model.encode(sample_description)

# Print the shape of the vector
print("Vector Dimension:", sample_vector.shape)