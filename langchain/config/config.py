'''
설정 파일 (모델 및 API 설정)
'''
# config.py
import torch



LLAMA_MODEL_PATH = "../models/converted-llama3.2-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# OLLAMA_API_URL = "http://localhost:11434/api/completion"
# config.py
OLLAMA_API_URL = "http://ollama:11434/"

# LangChain 관련 설정
LANGCHAIN_DEFAULT_PROMPT = "Write a response for the following input:"

KO_EN_MODEL_PATH = "../models/ko_en"
EN_KO_MODEL_PATH = "../models/en_ko"

MAX_LENGTH = 512

MILVUS_HOST = 'milvus-standalone'
MILVUS_PORT = '19530'

ETCD_ENDPOINTS = "etcd:2379"
MINIO_ADDRESS = "minio:9000"