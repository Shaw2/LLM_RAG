1. LangChain Folder에 들어온 상태에서 docker compose build

2. docker compose up

3. docker exec -it ollama bash

4. ollama pull llama3.2

4.1 ollama pull nomic-embed-text

5. 각 이미지별로 debug, info log message 확인 후 점검

6. milvus의 ip를 확인하여 docker inspect --format='{{json .NetworkSettings.Networks}}' milvus-standalone

7. python -> main.py connection 수정

7.1 python -> utils/helpers.py connection 수정

8. helpers.py 의 connection 수정


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