'''
메인 실행 파일
'''
# main.py
from pipelines.content_chain import ContentChain
from utils.helpers import languagechecker, insert_data, create_collection, search_data
from utils.ollama_embedding import get_embedding_from_ollama, get_embedding_from_ollama
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, List
from pydantic import BaseModel
from typing import Dict
from langchain.vectorstores import Milvus
from config.config import MILVUS_HOST, MILVUS_PORT
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection


app = FastAPI()
# ContentChain 초기화
content_chain = ContentChain()

# 요청 모델 정의
class GenerateRequest(BaseModel):
    input_text: str
    # is_korean: bool
    model: str = "llama3.2"  # Ollama 모델 이름, 기본값은 "default"
    
# API ENDPOINT 일괄처리 방식
@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    텍스트 생성 API 엔드포인트 (스트리밍 형태로 반환)
    """
    try:
        # 입력 텍스트가 한국어인지 판별
        discriminant = languagechecker(request.input_text)
        print(discriminant, "<===진행")

        # ContentChain에서 결과 생성
        result = content_chain.run(request.input_text, discriminant, model=request.model)
        print(f"Final result: {result}")  # 디버깅용 출력

        # 스트리밍 데이터 생성
        async def stream_response() -> AsyncGenerator[str, None]:
            for line in result.split("\n"):  # 결과를 줄 단위로 나눔
                yield line + "\n"  # 각 줄을 클라이언트에 스트리밍
                print(f"Streamed line: {line}")  # 디버깅용 출력

        # StreamingResponse로 반환
        return StreamingResponse(stream_response(), media_type="text/plain")

    except Exception as e:
        # 에러 발생 시 처리
        print(f"Error: {str(e)}")
        return {"error": str(e)}

# API 엔드포인트 streaming 방식
# @app.post("/generate")
# async def generate(request: GenerateRequest):
#     """
#     텍스트 생성 API 엔드포인트 (스트리밍 방식)
#     """
#     try:
#         discriminant = languagechecker(request.input_text)
#         print(discriminant, "<===진행")

#         # 스트리밍 데이터 생성
#         async def stream_response() -> AsyncGenerator[str, None]:
#             for chunk in content_chain.run(request.input_text, discriminant):
#                 yield chunk
#                 print(f"Yielded chunk: {chunk}")  # 디버깅용 출력

#         # StreamingResponse로 반환
#         return StreamingResponse(stream_response(), media_type="text/plain")

#     except Exception as e:
#         return {"error": str(e)}



# 헬스체크 엔드포인트
@app.get("/healthcheck")
async def healthcheck():
    """
    서버 상태 확인
    """
    return {"status": "ok"}



# ===========================================================================

# Milvus DB 사용 함수

# ===========================================================================

@app.get("/check")
def check_schema_db():
    connections.connect(host="172.19.0.6", port="19530")
    collection = Collection("ko_std_industry_collection")
    return collection.schema

@app.get("/api/db_connection")
def DB_connection():
    connections.connect(host="172.19.0.6", port="19530")
    collection = Collection("ko_std_industry_collection")
    return collection

@app.post("/api/db_create")    
def vectorDB_create(name: str = Body(...)):
    collection = create_collection(name)
    return {"status": "success", "collection_name": name}

class InsertRequest(BaseModel):
    question: str
    answer: str
    metadata: Dict[str, str]


@app.post('/api/insert')
def insert_data(request: InsertRequest):
    
    try:
        collection = Collection("ko_std_industry_collection")
        def process_text(text: str) -> List[float]:
            discriminant = languagechecker(text)
            if discriminant:
                translated_text = content_chain.ko_en_translator.translate(text)
                print(f"[DEBUG] Translated Text: {translated_text}")
                embedding = get_embedding_from_ollama(translated_text)
            else:
                print(f"[DEBUG] Original Text: {text}")
                embedding = get_embedding_from_ollama(text)

            if not embedding:
                raise ValueError("Failed to generate embedding: embedding is empty.")

            print(f"[DEBUG] Generated Embedding Length: {len(embedding)}")
            return embedding

        # 질문 및 답변 임베딩 처리
        question_embedding = process_text(request.question)
        answer_embedding = process_text(request.answer)

        # 임베딩 유효성 확인
        if not question_embedding or len(question_embedding) != 768:
            raise ValueError("Invalid question embedding dimension.")
        if not answer_embedding or len(answer_embedding) != 768:
            raise ValueError("Invalid answer embedding dimension.")

        # 데이터 준비
        questions = [request.question]
        answers = [request.answer]
        embeddings_question = [question_embedding]
        embeddings_answer = [answer_embedding]


        # metadata 데이터 추출
        metadata = request.metadata

        First_Category = [metadata.get("First_Category", "")]
        Second_Category = [metadata.get("Second_Category", "")]
        Third_Category = [metadata.get("Third_Category", "")]
        Fourth_Category = [metadata.get("Fourth_Category", "")]
        Fifth_Category = [metadata.get("Fifth_Category", "")]
        Menu = [metadata.get("Menu", "")]
        est_date = [metadata.get("est_date", "")]
        corp_name = [metadata.get("corp_name", "")]
        question_template = [metadata.get("question_template", "")]

        # 데이터 리스트 구성 (필드 순서에 맞게)
        data = [
            questions,
            answers,
            embeddings_question,
            embeddings_answer,
            First_Category,
            Second_Category,
            Third_Category,
            Fourth_Category,
            Fifth_Category,
            Menu,
            est_date,
            corp_name,
            question_template
        ]

        # 데이터 삽입
        collection.insert(data)

        # 인덱스 생성 (처음 한 번만 수행)
        if not collection.has_index():
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="embedding", index_params=index_params)

        # 컬렉션 로드
        collection.load()

        return {"status": "success", "message": "Data inserted successfully."}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert data: {str(e)}")




    
@app.post("/api/search")
def search_endpoint():
    # 컬렉션이 로드되어 있는지 확인하고, 로드되지 않았다면 로드
    collection = "ko_std_industry_collection"

    # 데이터 검색
    result = search_data(collection)
    return {"results": result}

@app.post("/api/test_search")
def search_test():
    collection_name = "test_collection"
    collection = Collection(collection_name)
    output_fields = ["question",
            "answer",
            "First_Category",
            "Second_Category",
            "Third_Category",
            "Fourth_Category",
            "Fifth_Category",
            "Menu",
            "est_date",
            "corp_name",
            "question_template"]

    results = collection.query(
        expr="",
        output_fields=output_fields,
        limit=100  # 최대 100개의 결과를 반환하도록 설정
    )

    return results

@app.post("/api/question_search")
def search_by_question(collection, query_text):
    def process_text(text: str) -> List[float]:
            discriminant = languagechecker(text)
            if discriminant:
                translated_text = content_chain.ko_en_translator.translate(text)
                print(f"[DEBUG] Translated Text: {translated_text}")
                embedding = get_embedding_from_ollama(translated_text)
            else:
                print(f"[DEBUG] Original Text: {text}")
                embedding = get_embedding_from_ollama(text)

            if not embedding:
                raise ValueError("Failed to generate embedding: embedding is empty.")

            print(f"[DEBUG] Generated Embedding Length: {len(embedding)}")
            return embedding

    query_embedding = process_text(query_text)
    output_fields = ["question",
            "answer",
            "First_Category",
            "Second_Category",
            "Third_Category",
            "Fourth_Category",
            "Fifth_Category",
            "Menu",
            "est_date",
            "corp_name",
            "question_template"]
    # 검색 파라미터 설정 (예: IVF_FLAT 인덱스 사용 시)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    results = collection.search(
        data=[query_embedding],
        anns_field="question_embedding",
        param=search_params,
        limit=3,
        expr=None,
        output_fields=output_fields
    )

    return results