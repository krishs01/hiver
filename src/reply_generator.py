"""
reply_generator.py — Generate grounded replies using RAG over historical responses.

Pipeline:
1. Embed historical brand replies using Gemini embedding API
2. For a new customer message, find similar past conversations via FAISS
3. Prompt LLM with retrieved context to draft a reply in the brand's voice

Citation:
  FAISS: https://github.com/facebookresearch/faiss
  Approach inspired by RAG (Retrieval-Augmented Generation) — Lewis et al., 2020
"""

import os
import json
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm


class ReplyGenerator:
    """RAG-based reply generator grounded in historical AppleSupport responses."""
    
    def __init__(self, data_path: str = None, index_path: str = None):
        """
        Initialize reply generator.
        
        Args:
            data_path: Path to processed conversation pairs CSV
            index_path: Directory to store/load FAISS index and embeddings
        """
        project_root = Path(__file__).resolve().parent.parent
        
        if data_path is None:
            data_path = project_root / "data" / "processed" / "apple_support_pairs.csv"
        if index_path is None:
            index_path = project_root / "data" / "embeddings"
        
        self.data_path = Path(data_path)
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.pairs_df = None
        self.index = None
        self.embeddings = None
        self._api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    def load_data(self):
        """Load conversation pairs."""
        self.pairs_df = pd.read_csv(self.data_path)
        print(f"[OK] Loaded {len(self.pairs_df):,} conversation pairs")
    
    def _embed_texts(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Embed a list of texts using Gemini embedding API."""
        from google import genai
        
        client = genai.Client(api_key=self._api_key)
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            # Clean empty strings
            batch = [t if t.strip() else "empty message" for t in batch]
            
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch,
            )
            
            for embedding in result.embeddings:
                all_embeddings.append(embedding.values)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def build_index(self, max_entries: int = 3000):
        """
        Build FAISS index from historical conversations.
        Embeds customer messages for similarity search.
        """
        import faiss
        
        if self.pairs_df is None:
            self.load_data()
        
        # Subsample for embedding (API cost management)
        if len(self.pairs_df) > max_entries:
            sample_df = self.pairs_df.sample(n=max_entries, random_state=42).reset_index(drop=True)
        else:
            sample_df = self.pairs_df.copy()
        
        self._indexed_df = sample_df
        
        # Check for cached index
        index_file = self.index_path / "faiss_index.bin"
        data_file = self.index_path / "indexed_data.pkl"
        
        if index_file.exists() and data_file.exists():
            print("[INFO] Loading cached FAISS index...")
            self.index = faiss.read_index(str(index_file))
            with open(data_file, "rb") as f:
                self._indexed_df = pickle.load(f)
            print(f"[OK] Loaded index with {self.index.ntotal} vectors")
            return
        
        # Embed customer messages
        texts = sample_df["customer_text_clean"].fillna("").tolist()
        print(f"[INFO] Embedding {len(texts)} customer messages...")
        self.embeddings = self._embed_texts(texts)
        
        # Build FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
        
        # L2 normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        # Cache
        faiss.write_index(self.index, str(index_file))
        with open(data_file, "wb") as f:
            pickle.dump(self._indexed_df, f)
        
        print(f"[OK] Built FAISS index with {self.index.ntotal} vectors (dim={dim})")
    
    def retrieve_similar(self, query: str, k: int = 5) -> list[dict]:
        """
        Find k most similar historical conversations to the query.
        
        Returns list of dicts with 'customer_text', 'brand_reply', 'score'.
        """
        import faiss
        
        if self.index is None:
            self.build_index()
        
        # Embed query
        query_embedding = self._embed_texts([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._indexed_df):
                continue
            row = self._indexed_df.iloc[idx]
            results.append({
                "customer_text": row["customer_text_clean"],
                "brand_reply": row["brand_text_clean"],
                "similarity_score": float(score),
            })
        
        return results
    
    def generate_reply(
        self,
        customer_message: str,
        intent: str = "",
        prior_context: str = "",
        n_examples: int = 5,
    ) -> dict:
        """
        Generate a reply grounded in historical responses.
        
        Args:
            customer_message: The customer's message
            intent: Classified intent (optional context)
            prior_context: Prior conversation context
            n_examples: Number of similar examples to retrieve
        
        Returns:
            dict with 'reply', 'retrieved_examples', 'reasoning'
        """
        from google import genai
        
        # Retrieve similar historical conversations
        similar = self.retrieve_similar(customer_message, k=n_examples)
        
        # Format retrieved examples for the prompt
        examples_text = ""
        for i, ex in enumerate(similar, 1):
            examples_text += f"\nExample {i} (similarity: {ex['similarity_score']:.2f}):\n"
            examples_text += f"  Customer: {ex['customer_text'][:200]}\n"
            examples_text += f"  Brand reply: {ex['brand_reply'][:200]}\n"
        
        context_section = ""
        if prior_context and prior_context.strip():
            context_section = f"\n[Prior conversation]: {prior_context}\n"
        
        intent_section = ""
        if intent:
            intent_section = f"\n[Detected intent]: {intent}\n"
        
        prompt = f"""You are an AI assistant drafting replies for Apple Support on Twitter.
Your replies should match Apple Support's actual tone and style: friendly, professional, 
empathetic, and concise (Twitter's character limit). Use the historical examples below 
as grounding for your response style and common resolution patterns.

GUIDELINES:
- Be empathetic and acknowledge the customer's frustration
- Provide specific, actionable next steps when possible
- Keep replies concise (under 280 characters ideally, max 2 tweets)
- Use Apple Support's typical language patterns from the examples
- If the issue requires account access or diagnostics, guide to DM or support link
- Never make up technical information — stick to what the examples show
{context_section}{intent_section}
HISTORICAL EXAMPLES OF HOW APPLE SUPPORT HANDLES SIMILAR ISSUES:
{examples_text}

CUSTOMER MESSAGE: {customer_message}

Draft a reply as Apple Support. Be specific to this customer's issue, not generic.
Return ONLY the reply text, no JSON, no explanation."""

        client = genai.Client(api_key=self._api_key)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        
        reply = response.text.strip()
        
        # Clean up: remove quotes if wrapped
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1]
        
        return {
            "reply": reply,
            "retrieved_examples": similar,
            "model": "gemini-2.0-flash",
        }
    
    def generate_reply_no_rag(self, customer_message: str, intent: str = "") -> dict:
        """
        Generate reply WITHOUT retrieval (for baseline comparison).
        Uses only the LLM's general knowledge.
        """
        from google import genai
        
        prompt = f"""You are Apple Support on Twitter. A customer sent this message:

"{customer_message}"

{f"Detected intent: {intent}" if intent else ""}

Write a helpful, empathetic reply as Apple Support. Keep it concise (Twitter length).
Return ONLY the reply text."""

        client = genai.Client(api_key=self._api_key)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        
        return {
            "reply": response.text.strip(),
            "retrieved_examples": [],
            "model": "gemini-2.0-flash (no RAG)",
        }


if __name__ == "__main__":
    gen = ReplyGenerator()
    
    if os.environ.get("GOOGLE_API_KEY"):
        print("Building index...")
        gen.build_index(max_entries=500)
        
        test_msg = "My iPhone keeps crashing after the iOS update, I've tried restarting"
        print(f"\nCustomer: {test_msg}")
        
        result = gen.generate_reply(test_msg, intent="SOFTWARE_UPDATE")
        print(f"\nGenerated reply: {result['reply']}")
        print(f"\nTop retrieved example:")
        if result["retrieved_examples"]:
            ex = result["retrieved_examples"][0]
            print(f"  Similar customer: {ex['customer_text'][:100]}")
            print(f"  How Apple replied: {ex['brand_reply'][:100]}")
    else:
        print("[SKIP] Set GOOGLE_API_KEY to test reply generation")
