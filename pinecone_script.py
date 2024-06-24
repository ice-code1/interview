import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone with your API key
pc = Pinecone(
    api_key='21f2127c-159b-4a3b-9fbc-a9b3a1ec422c'
)

# Load the cleaned dataset
df = pd.read_csv('cleaned_dataset.csv')

# Create an index
index_name = 'product-vectors'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Update dimension to match the vector dimension
        spec=ServerlessSpec(
            cloud='aws',
            region="us-east-1"  
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Initialize the SentenceTransformer model to create vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate product vectors and store them in Pinecone
def store_product_vectors(df, index, model):
    for idx, row in df.iterrows():
        product_id = str(row['StockCode'])

        if not all(ord(c) < 128 for c in product_id):
            continue  # Skip non-ASCII product IDs

        description = row['Description']
        vector = model.encode(description).tolist()  # Create vector from description
        metadata = {
            'Quantity': row['Quantity'],
            'UnitPrice': row['UnitPrice'],
            'CustomerID': row['CustomerID'],
            'Country': row['Country']
        }
        # Upsert the vector and metadata into Pinecone
        index.upsert([(product_id, vector, metadata)])

# Store the product vectors
store_product_vectors(df, index, model)

# Query the index with a sample vector
sample_description = "white metal lantern"
sample_vector = model.encode(sample_description).tolist()

# Retrieve the top 5 most similar vectors
query_result = index.query(queries=[sample_vector], top_k=5)

# Display the results
for match in query_result['matches']:
    print(f"Product ID: {match['id']}, Score: {match['score']}, Metadata: {match['metadata']}")



# Example product descriptions
description1 = "white metal lantern"
description2 = "set 7 babushka nesting boxes"

# Generate vectors
vector1 = model.encode(description1).tolist()
vector2 = model.encode(description2).tolist()

# Compute cosine similarity
cos_sim = cosine_similarity([vector1], [vector2])

print(f"Cosine Similarity between '{description1}' and '{description2}': {cos_sim[0][0]}")