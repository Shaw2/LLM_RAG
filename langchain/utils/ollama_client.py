import requests, json
from config.config import OLLAMA_API_URL

class OllamaClient:
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate'):
        self.api_url = api_url

    # 일괄 처리 방식
    def generate(self, model: str, prompt: str) -> str:
        """
        Ollama API를 사용하여 텍스트 생성
        Args:
            model (str): 사용할 Ollama 모델 이름
            prompt (str): 입력 프롬프트

        Returns:
            str: Ollama 모델의 생성된 텍스트
        """
        payload = {
            "model": model,
            "prompt": prompt
        }
        print("check payload")
        try:
            # Ollama API 요청
            print(self.api_url, payload, "response")
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 처리
            
            # JSON 응답 파싱
            full_response = response.text  # 전체 응답
            print("Full response content:", full_response)  # 디버깅용
            
            # 여러 JSON 객체가 하나의 문자열로 이어졌을 수 있으므로 분리
            lines = full_response.splitlines()
            all_text = ""
            for line in lines:
                try:
                    json_line = json.loads(line.strip())  # 각 줄을 JSON 파싱
                    all_text += json_line.get("response", "")
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON line skipped: {line}, Error: {e}")
                    continue

            return all_text.strip() if all_text else "Empty response received"

        except requests.exceptions.RequestException as e:
            print(f"HTTP 요청 실패: {e}")
            raise RuntimeError(f"Ollama API 요청 실패: {e}")

    # streaming 방식
    # def generate(self, model: str, prompt: str) -> str:
    #     """
    #     Ollama API를 사용하여 텍스트 생성
    #     Args:
    #         model (str): 사용할 Ollama 모델 이름
    #         prompt (str): 입력 프롬프트

    #     Returns:
    #         str: Ollama 모델의 생성된 텍스트
    #     """
    #     payload = {
    #         "model": model,
    #         "prompt": prompt
    #     }
    #     print("check payload")
    #     try:
    #         print(self.api_url, payload, "response")
    #         with requests.post(self.api_url, json=payload, stream=True) as response:
    #             response.raise_for_status()  # HTTP 에러 발생 시 예외 처리
    #             generated_text = ""
    #             for line in response.iter_lines(decode_unicode=True):
    #                 if line.strip():  # 빈 줄 제거
    #                     try:
    #                         print("Streamed line:", line)  # 디버깅용 출력
    #                         json_line = json.loads(line)
    #                         generated_text += json_line.get("response", "")
    #                     except json.JSONDecodeError:
    #                         print(f"Invalid JSON line skipped: {line}")
    #                         continue  # JSON 파싱 오류 시 건너뛰기
    #             return generated_text  # 전체 결과 반환
    #     except requests.exceptions.RequestException as e:
    #         raise RuntimeError(f"Ollama API 요청 실패: {e}")
    
    # 문장 단위 스트리밍 처리
    # def generate(self, model: str, prompt: str) -> str:
    #     payload = {
    #         "model": model,
    #         "prompt": prompt
    #     }
    #     try:
    #         print("Payload being sent:", payload)
    #         response = requests.post(self.api_url, json=payload, stream=True)
    #         response.raise_for_status()

    #         # 스트리밍 처리
    #         generated_text = ""
    #         sentence_buffer = ""
    #         for line in response.iter_lines(decode_unicode=True):
    #             if line.strip():
    #                 try:
    #                     print("Streamed line:", line)  # 디버깅용
    #                     json_line = json.loads(line.strip())  # JSON 파싱
    #                     chunk = json_line.get("response", "")

    #                     # 문장 완성 체크
    #                     sentence_buffer += chunk
    #                     while any(punct in sentence_buffer for punct in [".", "?", "!"]):  # 마침표 등 기준으로 문장 단위 분리
    #                         for punct in [".", "?", "!"]:
    #                             if punct in sentence_buffer:
    #                                 sentence, sentence_buffer = sentence_buffer.split(punct, 1)
    #                                 sentence += punct
    #                                 print("Completed sentence:", sentence.strip())  # 문장 단위 디버깅
    #                                 generated_text += sentence.strip() + "\n"
    #                 except json.JSONDecodeError as e:
    #                     print(f"Invalid JSON line skipped: {line}, Error: {e}")
    #                     continue

    #         # 마지막 남은 문장 추가
    #         if sentence_buffer.strip():
    #             generated_text += sentence_buffer.strip()

    #         return generated_text if generated_text else "Empty response received"

    #     except Exception as e:
    #         print(f"Error: {e}")
    #         raise RuntimeError(f"Ollama API 요청 실패: {e}")
