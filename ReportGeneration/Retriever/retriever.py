import os
import psycopg2
from dotenv import load_dotenv
from google import genai  # ✅ use Google GenAI for embeddings

# Load environment variables
load_dotenv()

# Initialize Google API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file.")

# Initialize Google GenAI client
client = genai.Client(api_key=api_key)

class ContextRetriever:
    def __init__(self):
        self.db_params = {
            "dbname": os.getenv("PG_DB"),
            "user": os.getenv("PG_USER"),
            "password": os.getenv("PG_PASSWORD"),
            "host": os.getenv("PG_HOST"),
            "port": os.getenv("PG_PORT"),
        }

    def _get_embedding(self, query: str):
        """
        Generate embedding for a query using Google Gemini Embedding API.
        """
        try:
            result = client.models.embed_content(
                model="models/embedding-001",
                contents=query
            )
            if result and hasattr(result, "embeddings"):
                return result.embeddings[0].values
            print("⚠️ Unexpected embedding format from Gemini API.")
            return None
        except Exception as e:
            print(f"❌ Error generating embedding from Gemini: {e}")
            return None

    def _connect_db(self):
        """
        Establish connection to Neon PostgreSQL database.
        """
        try:
            return psycopg2.connect(**self.db_params)
        except Exception as e:
            print(f"❌ Error connecting to Neon DB: {e}")
            return None

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve top_k most relevant chunks from Neon PostgreSQL pgvector table
        based on semantic similarity using Gemini embeddings.
        """
        query_vector = self._get_embedding(query)
        if not query_vector:
            return []

        conn = self._connect_db()
        if not conn:
            return []

        try:
            cursor = conn.cursor()

            # Convert vector to pgvector-compatible string
            vector_str = "[" + ", ".join(f"{x:.6f}" for x in query_vector) + "]"

            cursor.execute("""
                SELECT id, text, source, page
                FROM pdf_embeddings
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
            """, (vector_str, top_k))

            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            results = [
                {
                    "id": row[0],
                    "text": row[1],
                    "source": row[2],
                    "page": row[3],
                }
                for row in rows
            ]

            print(f"✅ Retrieved {len(results)} relevant results from Neon DB.")
            return results

        except Exception as e:
            print(f"❌ Error retrieving data from Neon DB: {e}")
            return []
