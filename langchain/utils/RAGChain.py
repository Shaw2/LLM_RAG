from langchain.prompts import PromptTemplate
from langchain.chains.base import Chain
from langchain.vectorstores import Milvus
from utils.ollama_client import OllamaClient  # Ollama API 호출

class CustomRAGChain(Chain):
    def __init__(self, retriever, llm, prompt_template):
        """
        Custom RAG Chain 생성자
        :param retriever: 문서 검색을 위한 retriever (Milvus)
        :param llm: 텍스트 생성 LLM (Ollama)
        :param prompt_template: LangChain에서 사용할 프롬프트 템플릿
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template

    @property
    def input_keys(self):
        return ["question"]

    @property
    def output_keys(self):
        return ["answer"]

    def _call(self, inputs):
        """
        Custom RAG Chain 처리 메서드
        :param inputs: 사용자 질문
        :return: Ollama에서 생성한 응답
        """
        question = inputs["question"]
        
        # 1. Milvus에서 질문에 대한 유사한 문서 검색
        docs = self.retriever.retrieve(question)  # Milvus에서 검색
        context = "\n\n".join([doc.page_content for doc in docs])  # 검색된 문서들 연결

        # 2. 프롬프트 템플릿 생성 (문서와 질문을 포함)
        prompt = self.prompt_template.format(context=context, question=question)

        # 3. Ollama API를 통해 텍스트 생성
        answer = self.llm._call(prompt)  # Ollama LLM 호출
        return {"answer": answer}  # 생성된 텍스트 반환
