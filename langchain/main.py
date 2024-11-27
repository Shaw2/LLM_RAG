'''
메인 실행 파일
'''
# main.py
from config.config import MILVUS_HOST, MILVUS_PORT
from pipelines.content_chain import ContentChain
from utils.helpers import languagechecker, insert_data, create_collection, search_data
from utils.ollama_embedding import get_embedding_from_ollama, get_embedding_from_ollama
from utils.ollama_client import OllamaClient
from utils.RAGChain import CustomRAGChain
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from langchain.vectorstores import Milvus
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pymilvus import connections, Collection, utility
from typing import AsyncGenerator, List
from pydantic import BaseModel
from typing import Dict


app = FastAPI()

connections.connect(host="172.19.0.6", port="19530")

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

@app.post("/api/generate_RAG")
async def generate_RAG(request: GenerateRequest):
    try:
        # 입력 텍스트
        input_text = request.input_text

        # 언어 감지
        discriminant = languagechecker(input_text)
        print(discriminant, "<===진행")

        # OllamaClient로 임베딩 생성 (언어에 따라 번역 포함)
        if discriminant:
            translated_text = content_chain.ko_en_translator.translate(input_text)
            query_embedding = get_embedding_from_ollama(translated_text)
        else:
            query_embedding = get_embedding_from_ollama(input_text)

        # Milvus 벡터 스토어 초기화 (컬렉션 이름, 임베딩 함수 및 연결 정보 제공)
        collection = Collection("ko_std_industry_collection")
        print(collection.describe())
        print(collection,"<=====connection")
        milvus_store = Milvus(
            collection_name="ko_std_industry_collection",
            embedding_function=get_embedding_from_ollama,  # 사용하려는 임베딩 함수
            connection_args={"host": "172.19.0.6", "port": "19530"}
        )
        print("milvus_store")
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        print("search_params")
        output_fields = ["question", "answer"]

        results = collection.search(
            data=[query_embedding],
            anns_field="question_embedding",
            param=search_params,
            limit=3,
            expr=None,
            output_fields=output_fields
        )

        print("results")
        print("milvus 관련 설정 완료")
        # 검색된 문서로 컨텍스트 구성
        retrieved_context = ""
        for hits in results:
            for hit in hits:
                retrieved_context += f"Q: {hit.entity.get('question')}\nA: {hit.entity.get('answer')}\n\n"

        print(f"Retrieved Context: {retrieved_context}")

        # Ollama API를 사용하여 최종 생성 수행
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Use the following context to answer the question:
            Context:
            {context}

            Question:
            {question}
            """
        )
        # OllamaClient로 LLM 호출
        ollama_llm = OllamaClient()  # OllamaClient 객체 생성
        print("ollama client 객체 생성 완료")

        # CustomRAGChain 생성
        custom_chain = CustomRAGChain(
            retriever=milvus_store.as_retriever(),
            llm=ollama_llm,  # OllamaClient를 LLM로 사용
            prompt_template=prompt_template
        )
        print("custom chain 생성 완료")
        # CustomRAGChain을 사용하여 최종 응답 생성
        response = custom_chain({"question": input_text})

        return {"response": response["answer"]}

    except Exception as e:
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
    collection = Collection("ko_std_industry_collection")
    return collection.schema

class DBCreateRequest(BaseModel):
    name: str

@app.post("/api/db_create")
def vectorDB_create(request: DBCreateRequest):
    collection = create_collection(request.name)
    return {"status": "success", "collection_name": request.name}

class InsertRequest(BaseModel):
    question: str
    answer: str
    metadata: Dict[str, str]
    

@app.post('/api/insert')
def insert_data(request: InsertRequest):
    
    try:
        collection_name = "ko_std_industry_collection"
        def check_collection_exists(collection_name: str) -> bool:
            """
            Checks if a collection exists in Milvus.
            
            Args:
                collection_name (str): The name of the collection to check.
            
            Returns:
                bool: True if the collection exists, False otherwise.
            """
            try:
                # Check if the collection exists
                return utility.has_collection(collection_name)
            except Exception as e:
                print(f"Error checking collection existence: {str(e)}")
                return False
        # Check if the collection exists, create if it doesn't
        if not check_collection_exists(collection_name):
            create_collection(collection_name)

        # Access the collection
        collection = Collection(collection_name)
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
        # 임베딩 유효성 확인
        if not question_embedding or len(question_embedding) != 768:
            raise ValueError("Invalid question embedding dimension.")

        # 데이터 준비
        questions = [request.question]
        answers = [request.answer]
        embeddings_question = [question_embedding]



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

        # # 인덱스 생성 (처음 한 번만 수행)
        # if not collection.has_index():
        #     index_params = {
        #         "index_type": "IVF_FLAT",
        #         "metric_type": "L2",
        #         "params": {"nlist": 128}
        #     }
        #     collection.create_index(field_name="question_embedding", index_params=index_params)

        # 컬렉션 로드
        collection.load()

        return {"status": "success", "message": "Data inserted successfully."}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert data: {str(e)}")


class SearchRequest(BaseModel):
    input_text: str
    collection: str  # 컬렉션 이름
    question: str  # 질문

@app.post("/api/search")
async def search_data(request: SearchRequest):
    try:
        # 입력 텍스트
        input_text = request.input_text
        collection_name = request.collection
        question = request.question

        # 언어 감지 및 임베딩 생성
        discriminant = languagechecker(input_text)
        if discriminant:
            translated_text = content_chain.ko_en_translator.translate(input_text)
            query_embedding = get_embedding_from_ollama(translated_text)
        else:
            query_embedding = get_embedding_from_ollama(input_text)

        # 컬렉션 접근
        collection = Collection(collection_name)

        # 검색 파라미터 설정
        search_params = {
            "metric_type": "L2",  # L2 거리 측정 방식 (유클리드 거리)
            "params": {"nprobe": 10}  # nprobe는 검색 정확도와 관련
        }

        # 검색
        results = collection.search(
            data=[query_embedding],  # 검색할 임베딩
            anns_field="question_embedding",  # 임베딩 필드
            param=search_params,
            limit=3,  # 상위 3개의 결과만 반환
            output_fields=["question", "answer"]  # 반환할 필드
        )

        # 검색 결과
        retrieved_context = ""
        for hits in results:
            for hit in hits:
                retrieved_context += f"Q: {hit.entity.get('question')}\nA: {hit.entity.get('answer')}\n\n"

        # 응답 반환
        return {"retrieved_context": retrieved_context}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}


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


class QuestionSearchRequest(BaseModel):
    collection: str
    question: str
    
@app.post("/api/question_search")
def search_by_question(request: QuestionSearchRequest):
    try:
        collection_name = request.collection
        question = request.question
        print(collection_name, question)

        collection = Collection(collection_name)

        # 텍스트 처리 및 임베딩 생성
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

        query_embedding = process_text(question)

        # 임베딩 유효성 확인
        if not query_embedding or len(query_embedding) != 768:
            raise ValueError("Invalid query embedding dimension.")

        # 검색 파라미터 설정 (예: IVF_FLAT 인덱스 사용 시)
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        output_fields = [
            "question",
            "answer",
            "First_Category",
            "Second_Category",
            "Third_Category",
            "Fourth_Category",
            "Fifth_Category",
            "Menu",
            "est_date",
            "corp_name",
            "question_template"
        ]

        # 검색 수행
        results = collection.search(
            data=[query_embedding],
            anns_field="question_embedding",
            param=search_params,
            limit=3,
            expr=None,
            output_fields=output_fields
        )

        # 검색 결과 처리
        result_data = []
        for hits in results:
            for hit in hits:
                result_data.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "question": hit.entity.get("question"),
                    "answer": hit.entity.get("answer"),
                    "First_Category": hit.entity.get("First_Category"),
                    "Second_Category": hit.entity.get("Second_Category"),
                    "Third_Category": hit.entity.get("Third_Category"),
                    "Fourth_Category": hit.entity.get("Fourth_Category"),
                    "Fifth_Category": hit.entity.get("Fifth_Category"),
                    "Menu": hit.entity.get("Menu"),
                    "est_date": hit.entity.get("est_date"),
                    "corp_name": hit.entity.get("corp_name"),
                    "question_template": hit.entity.get("question_template")
                })

        return {"results": result_data}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")