import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

class PineconeService:
    def __init__(self, api_key, index_name, dataset_filename):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dataset_filename = dataset_filename
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._initialize_index()

    def _initialize_index(self):
        df = pd.read_csv(self.dataset_filename)
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Update dimension to match the vector dimension
                spec=ServerlessSpec(
                    cloud='aws',
                    region="us-east-1"
                )
            )
            self._store_product_vectors(df)

    def _store_product_vectors(self, df):
        index = self.pc.Index(self.index_name)
        for idx, row in df.iterrows():
            product_id = str(row['StockCode'])

            if not all(ord(c) < 128 for c in product_id):
                continue  # Skip non-ASCII product IDs

            description = row['Description']
            vector = self.model.encode(description).tolist()
            metadata = {
                'Quantity': row['Quantity'],
                'UnitPrice': row['UnitPrice'],
                'CustomerID': row['CustomerID'],
                'Country': row['Country']
            }
            index.upsert([(product_id, vector, metadata)])

    def recommend_products(self, query_description):
        index = self.pc.Index(self.index_name)
        sample_vector = self.model.encode(query_description).tolist()
        query_result = index.query(queries=[sample_vector], top_k=5)

        matches = []
        for match in query_result['matches']:
            matches.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata']
            })
        return matches
