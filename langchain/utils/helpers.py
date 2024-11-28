import re
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np
from config.config import MILVUS_HOST, MILVUS_PORT
# from langchain.vectorstores import Milvus
from langchain_community.vectorstores import Milvus

def languagechecker(text):
    """
    입력된 문자열에 한글이 포함되어 있는지 확인합니다.
    
    Args:
        text (str): 검사할 문자열.
    
    Returns:
        bool: 한글이 포함되어 있으면 True, 아니면 False.
    """
    # 한글 유니코드 범위: ㄱ-ㅎ, ㅏ-ㅣ, 가-힣
    pattern = re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]')
    return bool(pattern.search(text))


# ==============================================================================================================

# MILVUS 함수

# ==============================================================================================================
# 컬렉션 생성 함수

connections.connect(host='172.19.0.6', port='19530')
def create_collection(name: str):
    try:
        collection_name = name

                # 컬렉션이 존재하면 삭제
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"Collection '{collection_name}' has been dropped.")

        # 스키마 정의
        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        question_field = FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1000)
        answer_field = FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=10000)
        question_embedding_field = FieldSchema(name="question_embedding", dtype=DataType.FLOAT_VECTOR, dim=768)

        # metadata 필드 추가
        metadata_fields = [
            FieldSchema(name="First_Category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="Second_Category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="Third_Category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="Fourth_Category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="Fifth_Category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="Menu", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="est_date", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="corp_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="question_template", dtype=DataType.VARCHAR, max_length=100)
        ]

        # 스키마 생성
        schema = CollectionSchema(
            fields=[id_field, question_field, answer_field, question_embedding_field] + metadata_fields,
            description="QA collection with metadata"
        )

        # 컬렉션 생성
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created successfully.")

        # 인덱스 생성 (처음 한 번만 수행)
        if not collection.has_index():
            index_params = {
                "index_type": "IVF_FLAT",  # 사용할 인덱스 유형
                "metric_type": "L2",       # 거리 측정 방식
                "params": {"nlist": 128}   # 인덱스 생성 파라미터
            }
            collection.create_index(field_name="question_embedding", index_params=index_params)  # 인덱스 생성
            print(f"Index created for 'question_embedding' field.")

        return collection

    except Exception as e:
        # 에러 발생 시 처리
        print(f"Error: {str(e)}")
        return {"error": str(e)}




# 데이터 삽입 함수
def insert_data(collection):
    # 임의의 벡터 데이터 생성
    data = [
        np.random.random(128).tolist() for _ in range(5)  # 5개의 128 차원 벡터
    ]
    
    # 데이터 삽입
    collection.insert([data])
    return "데이터가 삽입되었습니다."


def search_data(collection):
    # 컬렉션 로드 (이미 로드되었는지 확인하지 않아도 됩니다)
    collection.load()

    # 검색을 위한 임의의 벡터 생성
    query_vector = np.random.random(768).tolist()

    # 검색 수행
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=3,
        expr=None,
        output_fields=["question", "answer"]
    )

    # 결과 처리
    result_data = []
    for hits in results:
        for hit in hits:
            result_data.append({
                "id": hit.id,
                "distance": hit.distance,
                "question": hit.entity.get("question"),
                "answer": hit.entity.get("answer")
            })
    return result_data

# def collection_search(collection):
#     # 컬렉션 로드 (이미 로드되었는지 확인하지 않아도 됩니다)
#     collection.load()
#     res = collection.get_load_state()​
#     # 결과 처리
#     result_data = []
#     for hits in res:
#         for hit in hits:
#             result_data.append({
#                 "id": hit.id,
#                 "distance": hit.distance,
#                 "question": hit.entity.get("question"),
#                 "answer": hit.entity.get("answer")
#             })
#     return result_data