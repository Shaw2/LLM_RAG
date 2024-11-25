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
