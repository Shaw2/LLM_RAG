import requests, json, random
from config.config import OLLAMA_API_URL


class OllamaContentClient:
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.01, structure_limit = True):
        self.api_url = api_url
        self.temperature = temperature
        self.structure_limit = structure_limit
        
    async def send_request(self, model: str, prompt: str) -> str:
        """
        공통 요청 처리 함수: API 호출 및 응답 처리
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 처리

            full_response = response.text  # 전체 응답
            lines = full_response.splitlines()
            all_text = ""
            for line in lines:
                try:
                    json_line = json.loads(line.strip())  # 각 줄을 JSON 파싱
                    all_text += json_line.get("response", "")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    continue  # JSON 파싱 오류 시 건너뛰기
                
            return all_text.strip() if all_text else "Empty response received"

        except requests.exceptions.RequestException as e:
            print(f"HTTP 요청 실패: {e}")
            raise RuntimeError(f"Ollama API 요청 실패: {e}")
        
    async def landing_page_STD(self, model : str= "bllossom", input_text = "", section_cnt = 6):
        
        prompt = f"""
               <|start_header_id|>system<|end_header_id|>
                - 당신은 회사 소개 웹사이트의 기획자입니다.
                - Data는 회사 프로필 또는 회사 소개입니다.
                - 랜딩페이지를 만들때 섹션 흐름을 자연스럽게 구성해주세요.
                - Data는 아래 샘플 구조를 절대적으로 기반으로 생성되어야 합니다.
                - 추가 텍스트 없이 JSON 객체로만 답변하십시오.
                - 절대 code 데이터를 생성하지 마세요.
                - 입력 섹션 개수만큼 생성해주세요.
                - 섹션 개수를 정해진 만큼에서 넘어가면 안됩니다.
                - 섹션 개수는 9개를 초과할 수 없어요.
                - 1번은 헤더, 2번은 주로 히어로, 마지막은 푸터, 마지막 전은 [FAQ, 지도, 유튜브] 중에 하나로 추천해주세요.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                입력 데이터:
                {input_text}

                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                - 입력데이터를 기반으로 웹사이트 랜딩페이지 순서를 정해주세요.
                - 랜딩 페이지의 섹션 개수는{section_cnt}입니다. 
                - 1번은 헤더 섹션입니다.
                - 2번은 주로 히어로 혹은 메인비주얼 섹션입니다.
                - 마지막은 푸터 섹션입니다.
                - 섹션의 마지막 전에는 FAQ, Map, Youtube 중 하나를 추천해야합니다.
                - 나머지는 입력데이터를 랜딩페이지의 흐름 기준으로 적절히 알맞게 추천해주세요.
                - 섹션의 종류는 Introduce, Solution, Features, Social, Use Cases, Comparison, CTA, FAQ, Pricing, About Us, Team, Contact, Support, Newsletter, Subscription, Blog, Content 입니다.
                - 입력 데이터를 기반으로 웹사이트 랜딩 페이지 섹션을 하나의 JSON 객체로만 구성해주세요.
                - 결과 형식 예시는 다음과 같습니다:
                {{
                    "1": "Header",
                    "2": "Hero",
                    "3": "Features",
                    "4": "Content",
                    "5": "Pricing",
                    "6": "About Us",
                    "7": "Social",
                    "8": "Map",
                    "9": "Footer"
                }}
        """
        return await self.send_request(model, prompt)
    
    async def landing_block_STD(self, model : str= "bllossom", input_text = ""):
        
        prompt = f"""
               <|start_header_id|>system<|end_header_id|>
                
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                입력 데이터:
                {input_text}

                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                - 입력 데이터를 기반으로 웹사이트 랜딩 페이지 섹션을 하나의 JSON 객체로만 구성해주세요.
                - 결과 형식 예시는 다음과 같습니다:
    
        """
        return await self.send_request(model, prompt)
        
    async def LLM_land_page_content_Gen(self, input_text : str = "", model = "bllossom", structure_limit=True):
        # main landing page content Generate 하기 위해 필요한 코드가 여기에 들어가야함.
        # ------------------------------------------------
        cnt = random.randint(6,9)
        print(f"section count : {cnt}")
        try:
            # 비동기 함수 호출 시 await 사용
            section_data = await self.landing_page_STD(model=model, input_text=input_text, section_cnt=cnt)
            print(f"section_data: {section_data}")

            # JSON 문자열을 Python 딕셔너리로 변환
            section_dict = json.loads(section_data)
            print(f"type: {type(section_dict)},\n section_dict: {section_dict}")

            if structure_limit:
                # 문자열 키를 정수로 변환
                section_dict = {int(k): v for k, v in section_dict.items()}

                # 정렬된 키 리스트
                keys = sorted(section_dict.keys())
                footer_key = keys[-1]          # 마지막 키
                previous_key = keys[-2]        # 마지막 키 바로 이전

                # 1번 섹션 검증 및 수정
                if section_dict[1] != "Header":
                    section_dict[1] = "Header"

                # 2번 섹션 검증 및 수정
                if section_dict[2] != "Hero":
                    section_dict[2] = "Hero"

                # Footer 이전 섹션 검증 및 수정
                minus_one = ["FAQ", "Map", "Youtube"]
                minus_one_num = random.randint(0, 2)

                if section_dict[previous_key] not in minus_one:
                    section_dict[previous_key] = minus_one[minus_one_num]

                # Footer 섹션 검증 및 수정
                if section_dict[footer_key] != "Footer":
                    section_dict[footer_key] = "Footer"

            print(f"type: {type(section_dict)}, len section data: {len(section_dict)}")

            # 최종 수정된 딕셔너리를 JSON 문자열로 변환하여 반환
            return json.dumps(section_dict)
        except Exception as e:
            print(f"Error generating landing page sections: {e}")
            raise ValueError(status_code=500, detail="Failed to generate landing page sections.")

    async def LLM_land_block_content_Gen(self, input_text : str = "", model = "bllossom", section_num = "1", section_kind = ""):
        
        try:
            