import requests
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from config.config import OLLAMA_API_URL
import numpy as np
from langchain.embeddings.base import Embeddings

MAX_LENGTH = 512  # 모델에서 허용하는 최대 입력 길이 설정

class TextData(BaseModel):
    question: str
    answer: str
    metadata: dict

# def get_embedding_from_ollama(text: str) -> List[float]:
#     headers = {
#         "Content-Type": "application/json"
#     }

#     data = {
#         "model": "nomic-embed-text",  # 사용하려는 모델 이름
#         "input": text  # 임베딩할 텍스트
#     }
#     print(f"Input text: {text}")

#     # Ollama API에 POST 요청
#     try:
#         response = requests.post(OLLAMA_API_URL + 'api/embeddings', headers=headers, json=data)
#         print(response, "<=====response")

#         if response.status_code == 200:
#             response_data = response.json()
#             embedding = response_data.get("embedding", [])
#             print("Generated Embedding:", embedding)
#             return embedding  # 성공 시 임베딩 반환
#         else:
#             # 실패 시 로그와 기본값 반환
#             print("Error:", response.json())
#             return []  # 빈 리스트 반환
#     except requests.exceptions.RequestException as e:
#         # 요청 예외 처리
#         print(f"Request failed: {e}")
#         return []  # 요청 실패 시 빈 리스트 반환



def split_text(text: str, max_length: int = MAX_LENGTH) -> List[str]:
    """
    텍스트를 max_length 단위로 분할.
    """
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def get_embedding_from_ollama(text: str) -> List[float]: 
    headers = {"Content-Type": "application/json"} 
    if isinstance(text, list): 
        text = " ".join(map(str, text)) # 리스트의 각 요소를 문자열로 변환 후 합침 
    data = {"model": "nomic-embed-text", "prompt": text} 
    try: 
        response = requests.post(OLLAMA_API_URL + 'api/embeddings', headers=headers, json=data)
        response.raise_for_status() # 상태 코드가 4xx/5xx일 경우 예외 발생 
    except requests.RequestException as e: 
        print(f"[ERROR] Failed to get embedding from Ollama API: {str(e)}")
        raise 
    if response.status_code == 200: 
        embedding = response.json().get("embedding", []) 
        if not embedding: 
            print("[ERROR] Empty embedding received.") 
            raise ValueError("Failed to generate embedding: embedding is empty.") 
        print("[INFO] Embedding success") 
        return embedding 
    else: 
        error_message = response.json().get("error", "Unknown error") 
        print(f"[ERROR] Ollama API responded with error: {error_message}") 
        raise ValueError(f"Failed to get embedding from Ollama: {error_message}")

class OllamaEmbeddings(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        if isinstance(text, dict) and "query" in text:
            text = text["query"] # dict에서 query 키의 값을 추출 
        print(f"embed_query start {text} and type : {type(text)}") 
        return get_embedding_from_ollama(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [get_embedding_from_ollama(text) for text in texts]
