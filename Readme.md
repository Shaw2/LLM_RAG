1. LangChain Folder에 들어온 상태에서 docker compose build --no-cache
   (※ 모델 디렉토리를 langchain 경로에 놓고 build 진행)

2. docker compose up

3. 터미널을 새로 켜서 아래 명령어를 실행. {"status":"ok"}를 확인
   curl http://localhost:8000/healthcheck

4. 각 이미지별로 debug, info log message 확인 후 점검


--------

LLM - llama를 이용한 생성 방식 (한글 영어 상관 없음)
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{
           "input_text": "llama3.2 에 대해 알려줘",
           "model": "llama3.2"
         }'

LLM llama milvus 상태 체크
curl http://localhost:8000/healthcheck

LLM llama milvus의 db 생성
curl -X POST http://localhost:8000/api/db_create \
        -H "Content-Type: application/json" \
        -d '{
                "name": "ko_std_industry_collection"
            }'


http://localhost:8000/api/question_search?collection=ko_std_industry_collection&quert_text=트렌드알려줘

milvus 연결 체크
http://localhost:8000/api/db_connection

milvus insert 

curl -X POST http://localhost:8000/api/insert \
        -H "COntent-Type: application/json" \
        -d '{
                {
                    "question": "question 질문",
                    "answer": "answer for question 질문에 대한 답변",
                    "metadata": {
                        "First_Category": "도매 및 상품 중개업",
                        "Second_Category": "도매 및 상품 중개업",
                        "Third_Category": "상품 종합 도매업",
                        "Fourth_Category": "상품 종합 도매업",
                        "Fifth_Category": "상품 종합 도매업",
                        "Menu": "#main-content",
                        "est_date": "20160503",
                        "corp_name": "위인코리아_1",
                        "question_template": "business_question_template"
                        }
                }
            }'