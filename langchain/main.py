'''
메인 실행 파일
'''
# main.py
from config.config import MILVUS_HOST, MILVUS_PORT
from pipelines.content_chain import ContentChain
from utils.helpers import languagechecker, insert_data, create_collection, search_data
from utils.ollama_embedding import get_embedding_from_ollama, OllamaEmbeddings, embedding_from_ollama
from utils.ollama_client import OllamaClient, OllamaLLM
from utils.ollama_content import OllamaContentClient
from utils.ollama_chat import OllamaChatClient
from utils.RAGChain import CustomRAGChain
from utils.PDF2TXT import PDF2TEXT
from script.prompt import RAG_TEMPLATE, WEB_MENU_TEMPLATE

# local lib
# ------------------------------------------------------------------------ #
# outdoor lib

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from langchain.vectorstores import Milvus
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pymilvus import connections, Collection, utility
from typing import AsyncGenerator, List, Any, Union
from pydantic import BaseModel
from typing import Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests, json, re
from io import BytesIO
import time, random
import torch, gc

app = FastAPI()

# ContentChain 초기화
content_chain = ContentChain()

# 요청 모델 정의
class GenerateRequest(BaseModel):
    input_text: str
    # is_korean: bool
    model: str = "llama3.2"  # Ollama 모델 이름, 기본값은 "default"
    # model: str = "solar"  # Ollama 모델 이름, 기본값은 "default"
    gen_type: str = "normal"
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
        if request.gen_type == "normal":
            result = content_chain.run(request.input_text, discriminant, model=request.model, value_type=request.gen_type)
        if request.gen_type == "general":
            result = content_chain.run(request.input_text, discriminant, model=request.model, value_type=request.gen_type)
        print(f"Final result: {result}")  # 디버깅용 출력

        # 스트리밍 데이터 생성
        async def stream_response() -> AsyncGenerator[str, None]:
            for line in result.split("\n"):  # 결과를 줄 단위로 나눔
                yield line + "\n"  # 각 줄을 클라이언트에 스트리밍
                print(f"Streamed line: {line}")  # 디버깅용 출력

        # StreamingResponse로 반환
        # return StreamingResponse(stream_response(), media_type="text/plain")
        return {"responese" : result}

    except Exception as e:
        # 에러 발생 시 처리
        print(f"Error: {str(e)}")
        return {"error": str(e)}

@app.post("/api/generate_RAG")
async def generate_RAG(request: GenerateRequest):
    try:
        input_text = request.input_text

        # 언어 감지
        discriminant = languagechecker(input_text)

        # OllamaClient로 임베딩 생성 (언어에 따라 번역 포함)
        if discriminant:
            translated_text = content_chain.ko_en_translator.translate(input_text)
            query_embedding = get_embedding_from_ollama(translated_text)

        else:
            query_embedding = get_embedding_from_ollama(input_text)

        # Milvus 벡터 스토어 초기화 (컬렉션 이름, 임베딩 함수 및 연결 정보 제공)
        def collection_select():
            return
        collection = Collection("ko_std_industry_collection")
        try:
            # Milvus 연결 파라미터 확인
            milvus_store = Milvus(
                collection_name="ko_std_industry_collection",
                embedding_function=OllamaEmbeddings(),
                connection_args={
                    "host": "172.19.0.6",
                    "port": "19530"
                }, 
                vector_field="question_embedding", 
                text_field="question" # 문서의 텍스트 데이터를 나타내는 필드명을 question으로 설정
            )
            
            
            # Retriever 생성 전 추가 검증
            retriever = milvus_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 3}
            )

        except Exception as e:
            print("Milvus 초기화 중 오류 발생:")


        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        output_fields = ["question", "answer"]

        results = collection.search(
            data=[query_embedding],
            anns_field="question_embedding",
            param=search_params,
            limit=1,
            expr=None,
            output_fields=output_fields
        )


        # 검색된 문서로 컨텍스트 구성
        retrieved_context = ""
        for hits in results:
            for hit in hits:
                retrieved_context += f"Q: {hit.entity.get('question')}\nA: {hit.entity.get('answer')}\n\n"
        print(f"Retrieved context: {retrieved_context}")

        # 프롬프트 템플릿 정의
        template = ''
        def template_select():
            
            return template
        prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a web-making assistant specialized in web builders optimizing their platforms. For users using web builders, answer customized English questions.

        ## Context
        {context}

        ## Question
        {question}

        ### Instructions for Answer:
        - Respond in clear and precise English.
        - Focus on implementation details and practical steps.
        - Include recommendations for tools, frameworks, or best practices related to web building.
        - Suggest ways to enhance the user experience or improve platform efficiency.
        - Provide code snippets, if relevant, to illustrate technical solutions.
        - If applicable, outline a brief step-by-step process for implementing the solution.
        - I'm not asking you to generate code
        """
        )

        # OllamaClient로 LLM 호출
        ollama_client = OllamaClient()  # OllamaClient 객체 생성
        
        # LLM 래퍼 객체 생성
        # ollama_llm = OllamaLLM(client=ollama_client, model_name="llama3.2")
        ollama_llm = OllamaLLM(client=ollama_client, model_name="solar")

        # CustomRAGChain 생성
        custom_chain = CustomRAGChain(
            retriever=retriever,
            llm=ollama_llm,  # OllamaClient를 LLM로 사용
            prompt_template=prompt_template
        )
        
        # 동기 체인 호출을 비동기적으로 실행
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            response = await loop.run_in_executor(pool, lambda: custom_chain({"question": input_text}))
        
        # 번역 후 줄 바꿈 및 구조 유지
        def translate_line(line):
            if not line.strip():
                return ""
            translated =  content_chain.en_ko_translator.translate(line)
            return translated
        lines = response['answer'].split("\n")
        translated_with_linebreaks = "\n".join(
            translate_line(line) for line in lines
        )
        print(f"CustomRAGChain 응답: {translated_with_linebreaks}")
        return {"response": translated_with_linebreaks}

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
    
    

#---------------------------------
# Menu Generate Test
#---------------------------------

@app.post("/generate_menu")
async def generate_menu(path: str, path2: str='', path3: str=''):
    """
    텍스트 생성 API 엔드포인트 (스트리밍 형태로 반환)
    """
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        pdf_list = []
        response = requests.get(path)

        if response.status_code == 200:
            pdf_data = BytesIO(response.content)
            pdf_list.append(pdf_data)
            
            if path2 != ''  :
            # print("path2 : ", path2)
                response2 = requests.get(path2)
                pdf_data2 = BytesIO(response2.content)
                pdf_list.append(pdf_data2)
            if path3 != '' :
            # print("path3 : ", path3)
                response3 = requests.get(path3)
                pdf_data3 = BytesIO(response3.content)
                pdf_list.append(pdf_data3)
        all_text = PDF2TEXT(pdf_list)
        print(all_text[:1000], len(all_text),"<====all_text \n")
  
        
        start = time.time() 
        # 입력 텍스트가 한국어인지 판별 뺴야함 이부분들 한국어 체크
        discriminant = languagechecker(all_text)
        if discriminant:
            if len(all_text) > 2500:
                all_text = all_text[:2500]
        else:
            if len(all_text) > 8192:
                all_text = all_text[:8192]
        
        print(len(all_text), "after")
        print(discriminant, "<===진행")

        # ContentChain에서 결과 생성
        # result = content_chain.run(all_text, discriminant, model='llama3.2', value_type='menu')
        result = content_chain.run(all_text, discriminant, model='solar', value_type='menu')
        print("process end structure")
        # if result['menu_structure']:
        #     menu_content = []
        #     for menu in result['menu_structure']:
        #         context = None
        #         menu_content_dict = {}
        #         context = content_chain.contents_run(model='solar', input_text=all_text, menu=menu)
        #         menu_content_dict[menu] = context
        #         menu_content.append(menu_content_dict)
        #         print(f"menu_content_dict[menu] : {menu_content_dict[menu]}")
        #     result['menu_content'] = menu_content
                
        
        print(f"Final result: {result}")  # 디버깅용 출력

        end = time.time()
        
        print("process time : ", end - start)
        return result

    except Exception as e:
        # 에러 발생 시 처리
        print(f"Error: {str(e)}")
        return {"error": str(e)}
    

class LandPageRequest(BaseModel):
    input_text: str
    model: str = "solar"
    
# 엔드포인트 정의
@app.post("/generate_land_section")
async def LLM_land_page_generate(request: LandPageRequest):
    """
    랜딩 페이지 섹션 생성을 처리하는 API 엔드포인트
    """
    
    try:
        print(f"Start process: {request.input_text}")
        print(f"Model: {request.model}")
        
        # OllamaContentClient와 ConversationHandler 초기화
        content_client = OllamaContentClient()
        
        # content = await content_client.contents_GEN(input_text=request.input_text, section_name="")
        content = await content_client.contents_GEN_chat(input_text=request.input_text, section_name="")
        
        return content
        
        # # STEP 1: 랜딩 페이지 섹션 구조 생성
        # section_options = ["Introduce", "Solution", "Features", "Social", "CTA", "Pricing", "About Us", "Team", "blog"]
        # section_cnt = random.randint(6, 9)
        # print(f"Selected section count: {section_cnt}")

        # # 섹션 고정 및 랜덤 채움
        # section_dict = {1: "Header", 2: "Hero", section_cnt - 1: random.choice(["FAQ", "Map", "Youtube", "Contact", "Support"]), section_cnt: "Footer"}
        # filled_indices = {1, 2, section_cnt - 1, section_cnt}
        # for i in range(3, section_cnt):
        #     if i not in filled_indices:
        #         section_dict[i] = random.choice(section_options)
        # landing_structure = dict(sorted(section_dict.items()))
        
        # print(f"Generated landing structure: {landing_structure}")
        # # content = await content_client.contents_GEN(input_text=request.input_text)
        # # print(f"Generated content : {content}")

        # main_dict = {"text": "", "type": "LANDING", "children": []}

        # for section_num, section_name in landing_structure.items():
        #     print(f"Processing section {section_num}: {section_name}")

        #     time.sleep(0.5)
        #     content = await content_client.contents_GEN(input_text=request.input_text, section_name=section_name)
        #     content = content.replace("<|start_header_id|>", "").replace("<|end_header_id|>", "").replace("<|eot_id|>", "")
            
        #     print(f"content : {content}")
        #     data_dict = json.loads(content)

        #     # 여기서 main_dict["children"] 대신 data_dict["children"]로부터 type 추출
        #     tag_list = [child['type'] for child in data_dict["children"] if 'type' in child]
        #     tag = "_".join(tag_list)

        #     data_dict["tag"] = tag
        #     data_dict["section_name"] = section_name

        #     # 이제 main_dict에 data_dict를 append
        #     main_dict["children"].append(data_dict)


        # return main_dict

    except Exception as e:
        print(f"Error processing landing structure: {e}")
        raise HTTPException(status_code=500, detail="Error processing landing structure.")
    
    # try:
    #     print(f"Start process: {request.input_text}")
    #     print(f"Model: {request.model}")
        
    #     # OllamaContentClient와 ConversationHandler 초기화
    #     content_client = OllamaContentClient()

    #     # STEP 1: 랜딩 페이지 섹션 구조 생성
    #     section_options = ["Introduce", "Solution", "Features", "Social", "CTA", "Pricing", "About Us", "Team", "blog"]
    #     section_cnt = random.randint(6, 9)
    #     print(f"Selected section count: {section_cnt}")

    #     # 섹션 고정 및 랜덤 채움
    #     section_dict = {1: "Header", 2: "Hero", section_cnt - 1: random.choice(["FAQ", "Map", "Youtube", "Contact", "Support"]), section_cnt: "Footer"}
    #     filled_indices = {1, 2, section_cnt - 1, section_cnt}
    #     for i in range(3, section_cnt):
    #         if i not in filled_indices:
    #             section_dict[i] = random.choice(section_options)
    #     landing_structure = dict(sorted(section_dict.items()))
        
    #     print(f"Generated landing structure: {landing_structure}")
    #     # content = await content_client.contents_GEN(input_text=request.input_text)
    #     # print(f"Generated content : {content}")

    #     main_dict = {"text": "", "type": "LANDING", "children": []}

    #     for section_num, section_name in landing_structure.items():
    #         print(f"Processing section {section_num}: {section_name}")

    #         time.sleep(0.5)
    #         content = await content_client.contents_GEN(input_text=request.input_text, section_name=section_name)
    #         content = content.replace("<|start_header_id|>", "").replace("<|end_header_id|>", "").replace("<|eot_id|>", "")
            
    #         print(f"content : {content}")
    #         data_dict = json.loads(content)

    #         # 여기서 main_dict["children"] 대신 data_dict["children"]로부터 type 추출
    #         tag_list = [child['type'] for child in data_dict["children"] if 'type' in child]
    #         tag = "_".join(tag_list)

    #         data_dict["tag"] = tag
    #         data_dict["section_name"] = section_name

    #         # 이제 main_dict에 data_dict를 append
    #         main_dict["children"].append(data_dict)


    #     return main_dict

    # except Exception as e:
    #     print(f"Error processing landing structure: {e}")
    #     raise HTTPException(status_code=500, detail="Error processing landing structure.")


def extract_json_from_response(response_text: str) -> Union[dict, None]:
    """
    주어진 텍스트에서 JSON 형식을 찾아 반환합니다.
    """
    try:
        # '{'로 시작해서 '}'로 끝나는 모든 JSON 형식을 찾음
    
        json_pattern = r"(\{.*?\})"
        matches = re.findall(json_pattern, response_text, re.DOTALL)

        for match in matches:
            try:
                # JSON 변환 시도
                parsed_json = json.loads(match)
                # 필수 키 검증
                if "children" in parsed_json:
                    return parsed_json
            except json.JSONDecodeError:
                continue  # JSON 파싱 실패 시 다음으로 넘어감

        print("No valid JSON structure found.")
        return None
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return None

# @app.post("/generate_land_section")    
# async def LLM_land_page_generate(input_text: str = "", model="solar", structure_limit=True):
#     # Main landing page content generation
#     content_LLM = OllamaContentClient()
#     landing_structure = await content_LLM.LLM_land_page_content_Gen(input_text=input_text, model=model, structure_limit=structure_limit)
#     print(f"landing_structure : {landing_structure}")
#     try:
#         # JSON 문자열을 딕셔너리로 변환
#         if isinstance(landing_structure, str):
#             landing_structure = json.loads(landing_structure)
#         summary = content_LLM.LLM_summary(model=model, input_text=input_text)
#         # 딕셔너리 반복 처리
#         for k, v in landing_structure.items():
#             contents_data = await content_LLM.LLM_land_block_content_Gen(
#                 model=model,
#                 input_text=input_text,
#                 section_name=v,
#                 section_num=k,
#                 summary = summary
#             )
#             print(f"keys : {k}, value : {v}\n contents_data: {contents_data}")
        
#         return landing_structure
#     except json.JSONDecodeError as e:
#         print(f"Invalid JSON format: {landing_structure}")
#         raise HTTPException(status_code=500, detail="Failed to parse JSON from LLM response.")
#     except Exception as e:
#         print(f"Error processing landing structure: {e}")
#         raise HTTPException(status_code=500, detail="Error processing landing structure.")

# Chat Test

@app.post("/chat_landpage_generate")
async def chat_landpage_generate(request: LandPageRequest):
    client = OllamaChatClient()
    section_options = ["Introduce", "Solution", "Features", "Social", "CTA", "Pricing", "About Us", "Team", "blog"]
    section_cnt = random.randint(6, 9)
    print(f"Selected section count: {section_cnt}")

    await client.store_chunks(model=request.model, data=request.input_text)
    
    # 섹션 고정 및 랜덤 채움
    section_dict = {1: "Header", 2: "Hero", section_cnt - 1: random.choice(["FAQ", "Map", "Youtube", "Contact", "Support"]), section_cnt: "Footer"}
    filled_indices = {1, 2, section_cnt - 1, section_cnt}
    for i in range(3, section_cnt):
        if i not in filled_indices:
            section_dict[i] = random.choice(section_options)
    landing_structure = dict(sorted(section_dict.items()))
    
    
    print(f"Generated landing structure: {landing_structure}")
    for section_num, section_name in landing_structure.items():
            print(f"Processing section {section_num}: {section_name}")

            time.sleep(0.5)
            # content = await content_client.generate_section(input_text=request.input_text, section_name=section_name)
            generated_content = await client.generate_section(model=request.model, section_name=section_name)
            print(f"content : {generated_content}")

    return generated_content