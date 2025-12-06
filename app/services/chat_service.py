"""
Service untuk RAG Chatbot menggunakan OpenAI
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI


class ChatService:
    def __init__(self, model_transformer: SentenceTransformer, 
                 df: pd.DataFrame, issue_embeddings: np.ndarray):
        self.model_transformer = model_transformer
        self.df = df
        self.issue_embeddings = issue_embeddings
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
    
    def retrieve_context(self, query: str, top_k: int = 3) -> tuple[str, List[Dict]]:
        """
        Retrieve relevant context dari knowledge base
        Returns: (context_text, sources)
        """
        # Translate query to English
        english_query = self._translate_query(query)
        
        # Encode query (force CPU untuk consistency)
        query_embedding = self.model_transformer.encode(english_query, convert_to_tensor=True)
        
        # Pastikan semua tensor di CPU
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu()
        
        # Ensure embeddings are numpy arrays, then convert to CPU tensor
        if isinstance(self.issue_embeddings, torch.Tensor):
            embeddings_tensor = self.issue_embeddings.cpu()
        else:
            embeddings_tensor = torch.tensor(self.issue_embeddings, device='cpu')
        
        # Semantic search
        cos_scores = util.pytorch_cos_sim(query_embedding, embeddings_tensor)[0]
        
        # Get top-K
        top_results = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))
        
        retrieved_data = []
        sources = []
        
        for score, idx in zip(top_results.values, top_results.indices):
            idx = int(idx)
            row = self.df.iloc[idx]
            
            methods = str(row['methods']) if isinstance(row['methods'], list) else "[]"
            context_str = (
                f"Ticket: {row['key']}\n"
                f"Summary: {row['summary']}\n"
                f"Technical Methods involved: {methods}\n"
                f"Relevance Score: {score:.2f}\n"
            )
            retrieved_data.append(context_str)
            
            sources.append({
                "key": row['key'],
                "summary": row['summary'],
                "score": float(score)
            })
        
        context_text = "\n---\n".join(retrieved_data)
        return context_text, sources
    
    def _translate_query(self, query: str) -> str:
        """Translate user query to technical English"""
        translation_prompt = f"""
        Translate the following user question into a clear, technical English search query for a software issue database.
        Only output the English translation, nothing else.
        
        User Question: "{query}"
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": translation_prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return query  # Fallback to original query
    
    def chat(self, user_query: str, context_text: str) -> str:
        """Generate response using LLM with context"""
        system_prompt = """
        You are Knowledge Hub, a polite and helpful Senior Software Engineer Assistant. 
        You have access to a database of historical software issues (Jira tickets).
        
        Instructions:
        1. Identity: If the user asks "Who are you?" or "What is your name?", you MUST answer: "I am Knowledge Hub". Do not use any other name.
        2. If the user greets or chats casually (e.g., "Halo", "Apa kabar"), just reply naturally and politely in the user's language. Ignore the context.
        3. If the user asks a technical question, answer it based ONLY on the provided Context.
        4. The Context is in English, but the user might ask in Indonesian. You MUST translate the insight and answer in the user's language.
        5. If the context contains specific Java methods/classes, mention them exactly as they are (do not translate code).
        6. Be concise and professional.
        """
        
        user_message = f"""
        Context from Knowledge Base:
        {context_text}
        
        User Question: 
        {user_query}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def process_query(self, query: str, top_k: int = 3) -> Dict:
        """
        Main method untuk process user query
        Returns: {"answer": str, "sources": List[Dict]}
        """
        context_text, sources = self.retrieve_context(query, top_k)
        answer = self.chat(query, context_text)
        
        return {
            "answer": answer,
            "sources": sources
        }
