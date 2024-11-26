1. LangChain Folder에 들어온 상태에서 docker compose build
   (※ 모델 디렉토리를 langchain 경로에 놓고 build 진행)

2. docker compose up

3. 터미널을 새로 켜서 아래 명령어를 사용하려 milvus의 ip를 확인
   docker inspect --format='{{json .NetworkSettings.Networks}}' milvus-standalone
   or
   docker inspect milvus-standalone
   
4. 아래 스크립트 들에서 ip 수정
   4-1. python -> main.py connection에서 ip 수정
   4-2. python -> utils/helpers.py connection에서 ip 수정

5. docker exec -it ollama bash

6. ollama pull llama3.2

6.1 ollama pull nomic-embed-text

7. 터미널을 새로 켜서 아래 명령어를 실행. {"status":"ok"}를 확인
   curl http://localhost:8000/healthcheck

8. 각 이미지별로 debug, info log message 확인 후 점검



