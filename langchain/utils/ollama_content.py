import requests, json, random, re
from config.config import OLLAMA_API_URL
from fastapi import HTTPException
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import tiktoken
from typing import List
from utils.ollama_embedding import get_embedding_from_ollama

class OllamaContentClient:
    def __init__(self, api_url=OLLAMA_API_URL+'api/chat', temperature=0, structure_limit = True,  n_ctx = 4196, max_token = 4196):
        self.api_url = api_url
        self.temperature = temperature
        self.structure_limit = structure_limit
        self.n_ctx = n_ctx
        self.max_token = max_token
        
    async def send_request(self, model: str, prompt: str) -> str:
        """
        공통 요청 처리 함수: API 호출 및 응답 처리
        """
        
        payload = {
            "model": model,
            "message": prompt,
            "temperature": self.temperature,
            "n_ctx": self.n_ctx,
            "repetition penalty":1.2,
            "stream" : False
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 처리

            full_response = response.text  # 전체 응답
            lines = full_response.splitlines()
            print("lines : ", lines)
            
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
        
    async def contents_GEN_chat(self, model : str= "solar", input_text = "", section_name=""):

        message = [
                        {
                        "role": "system",
                        "content" : "너는 단일 페이지로 구성된 랜딩 사이트에 들어갈 내용을 작성해주는 AI 도우미야.",
                        "role" : "user",
                        "content" : input_text
                        }
                    ]
        response_user = await self.send_request(model, [message])
        print(f"User response: {response_user}")
        
        return response_user.strip()        
        # return await self.send_request(model, message)
    
    async def contents_GEN(self, model : str= "solar", input_text = "", section_name=""):
                # 1. assistant처럼 생성해야하고 형식을을 **절대** 벗어나면 안 된다.
                # 6. 출력 결과는 코드 형태만 허용된다. 코드는 **절대 생성하지 마라.**
                # 7. 오직 한글로만 작성하라.
                # - 섹션의 갯 수는 PDF 데이터의 구성을 고려해서 결정해줘.
                # - 입력된 PDF 데이터의 구성과 섹션의 순서를 함께 고려해서, 섹션에 들어갈 수 있게 데이터를 만들어줘.
                # 2. "html, body, header, footer, section" 태그는 사용하지 않는다.

        prompt = f""" 
                <|start_header_id|>system<|end_header_id|>
                - 너는 단일 페이지로 구성된 랜딩 사이트에 들어갈 내용을 작성해주는 AI 도우미야.
                - 섹션은 6~10개 사이로 자유롭게 구성해줘. 대신 랜딩페이지의 기본적인 골격에 벗어나지 않아야 해.
                - 섹션에 어울리는 내용을 생성해야 하며, 반드시 다음 규칙을 따라야 한다:
                    1. "div, h1, h2, h3, p, ul, li" 태그만 사용해서 섹션의 콘텐츠를 구성한다.
                    2. 섹션 안의 컨텐츠 개수는 2~10개 사이에서 자유롭게 선택한다.
                    3. '생성 예시'를 그대로 사용하지 않으며, 사용자 input을 고려하여 내용이 반복되지 않도록 다양하게 생성한다.
                    4. 모든 텍스트 내용은 입력 데이터에 맞게 작성하고, 섹션의 목적과 흐름에 맞춰야 한다.
                    5. 태그를 제외한 내용은 오직 한글로만 작성한다.
                    
                - 한글로 작성했는지 다시 한 번 확인해서. 한글로 출력한다.
                
                **생성 예시**
                '''
                <div class="HEADER_SECTION">                                                                                                                                     
                    <h1>회사 이름 혹은 제품 이름</h1>
                    <p>회사 소개 혹은 제품 소개</p>
                </div>																																						    
                <div class="CTA_SECTION">                                                                                                                               
                    <h1>서비스 소개</h1>
                    <p><a href="">바로 가기</a></p>
                </div>
                <div class="FEATURES_SECTION">
                    <h3>서비스 강점</h3>
                    <h1>서비스 주요 특징</h1>
                    <p>서비스 주요 특징 설명</p>
                    <ul>                                                                                                                                 
                        <li>
                            <h2>특징 1</h2>
                            <p>특징 1에 대한 설명</p>
                        </li>
                        <li>
                            <h2>특징 2</h2>
                            <p>특징 2에 대한 설명</p>
                        </li>
                        <li>
                            <h2>특징 3</h2>
                            <p>특징 3에 대한 설명</p>
                        </li>
                    </ul>
                </div>
                <div class="PRICING_SECTION">                                                                                                                   
                    <h3>가격 경쟁력 포인트</h3>
                    <p>서비스 가격 소개</p>
                </div>
                <div class="FOOTER_SECTION">                                                                                                                                
                    <h1>서비스 다시 소개</h1>
                    <p><a href="">바로 가기</a></p>
                </div>
                '''

                <|eot_id|><|start_header_id|>user<|end_header_id|>
                입력 데이터:
                {input_text}
                섹션:
                {section_name}


                <|eot_id|><|start_header_id|>assistant<|end_header_id|>

                """


        # prompt = f"""
        #         <|start_header_id|>system<|end_header_id|>
        #         - 너는 사이트의 섹션 구조를 정해주고, 그 안에 들어갈 내용을 작성해주는 AI 도우미야.
        #         - 입력된 데이터를 기준으로 단일 페이지를 갖는 랜딩사이트 콘텐츠를 생성해야 해.
        #         - 'children'의 컨텐츠 내용의 수는 너가 생각하기에 섹션에 알맞게 개수를 수정해서 생성해줘.
        #         - 섹션 '{section_name}'에 어울리는 내용을 생성해야 하며, 반드시 다음 규칙을 따라야 한다:
                
        #         1. "div, h1, h2, h3, p, ul, li" 태그만 사용해서 섹션의 콘텐츠를 구성해라.
        #         2. "html, body, header, footer, section" 태그는 사용하지 않는다.
        #         3. 섹션 안의 `children` 안의 컨텐츠 개수는 2~10개 사이에서 자유롭게 선택하되, 내용이 반복되지 않도록 다양하게 생성하라.
        #         4. 모든 텍스트 내용은 입력 데이터에 맞게 작성하고, 섹션의 목적과 흐름에 맞춰야 한다.
        #         5. 태그를 제외한 내용은 오직 한글로만 작성하라.
                
        #         **생성 예시**
        #         ```
        #             <div class="HEADER_SECTION">
        #             <h2>Our Eco-Friendly Products</h2>
        #             <ul>
        #                 <li><a href="#solar-panels">Solar Panels</a></li>
        #                 <li><a href="#energy-efficient-appliances">Energy Efficient Appliances</a></li>
        #                 <li><a href="#water-saving-devices">Water Saving Devices</a></li>
        #             </ul>
        #             <div class="CTA_SECTION">
        #                 <!-- Solar Panels Section -->
        #                 <div id="solar-panels">
        #                 <h3>Solar Panels</h3>
        #                 <p>Harness the power of the sun with our high-quality solar panels. Not only do they reduce your energy
        #             bills, but they also minimize your carbon footprint by producing clean, renewable energy.</p>
        #                 </div>
        #                 <!-- Energy Efficient Appliances Section -->
        #                 <div id="energy-efficient-appliances">
        #                 <h3>Energy Efficient Appliances</h3>
        #                 <p>Upgrade your appliances with our energy-efficient options. They not only save you money on your utility
        #             bills but also help conserve natural resources and reduce greenhouse gas emissions.</p>
        #                 </div>
        #                 <!-- Water Saving Devices Section -->
        #                 <div id="water-saving-devices">
        #                 <h3>Water Saving Devices</h3>
        #                 <p>Save water and money with our water-saving devices. They help conserve this precious resource and reduce
        #             the strain on local water systems, all while keeping your bills low.</p>
        #                 </div>
        #             </div>
        #             </div>
        #         ```

        #         <|eot_id|><|start_header_id|>user<|end_header_id|>
        #         입력 데이터:
        #         {input_text}
        #         섹션:
        #         {section_name}
                

        #         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                
                
        #         """
        # prompt = f"""
        #         <|start_header_id|>system<|end_header_id|>
        #         - 너는 사이트의 섹션 구조를 정해주고, 그 안에 들어갈 내용을 작성해주는 AI 도우미야.
        #         - 입력된 데이터를 기준으로 단일 페이지를 갖는 랜딩사이트 콘텐츠를 생성해야 해.
        #         - 'children'의 컨텐츠 내용의 수는 너가 생각하기에 섹션에 알맞게 개수를 수정해서 생성해줘.
        #         - 섹션 '{section_name}'에 어울리는 내용을 생성해야 하며, 반드시 다음 규칙을 따라야 한다:
        #         1. assistant처럼 생성해야하고 형식을을 **절대** 벗어나면 안 된다.
        #         2. "h1, h2, h3, p" 태그만 사용해서 섹션의 콘텐츠를 구성해라.
        #         3. 섹션 안의 `children` 안의 컨텐츠 개수는 2~10개 사이에서 자유롭게 선택하되, 내용이 반복되지 않도록 다양하게 생성하라.
        #         4. 모든 텍스트 내용은 입력 데이터에 맞게 작성하고, 섹션의 목적과 흐름에 맞춰야 한다.
        #         5. 출력 결과는 JSON 형태만 허용된다. 코드는 **절대 생성하지 마라.**
        #         6. 오직 한글로만 작성하라.
                

        #         <|eot_id|><|start_header_id|>user<|end_header_id|>
        #         입력 데이터:
        #         {input_text}
        #         섹션:
        #         {section_name}
                

        #         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        #         - 너는 JSON 형태의 응답만을 반환해야 한다. 아래와 같은 형식의 순수 JSON만을 출력해야해.
        #         {{
        #             "children": [
        #                 {{
        #                     "type": "h1",
        #                     "text": "섹션의 주요 제목을 입력 데이터에 맞게 작성합니다."
        #                 }},
        #                 {{
        #                     "type": "p",
        #                     "text": "섹션의 내용을 소개하는 첫 번째 단락입니다. 핵심 메시지를 간결하고 명확하게 작성합니다."
        #                 }},

        #                 {{
        #                     "type": "h3",
        #                     "text": "중요 포인트를 정리하거나 강조합니다."
        #                 }},
        #                 {{
        #                     "type": "p",
        #                     "text": "설명 내용을 구체적으로 작성하되 중복되지 않도록 주의합니다."
        #                 }},
        #                 {{
        #                     "type": "p",
        #                     "text": "마무리 문장으로 섹션의 가치를 강조하고 독자의 참여를 유도합니다."
        #                 }}
        #             ]
        #         }}
        #         """
        return await self.send_request(model, prompt)
    
    async def landing_block_STD(self, model : str= "solar", input_text :str = "", section_name=""):
        prompt = f"""
            <|start_header_id|>system<|end_header_id|>
            - 당신은 AI 랜딩페이지 콘텐츠 작성 도우미입니다.
            - 입력된 데이터를 기반으로 랜딩페이지의 적합한 콘텐츠를 작성하세요.
            - 반드시 입력 데이터를 기반으로 작성하며, 추가적인 내용은 절대 생성하지 마세요.
            - 섹션에 이름에 해당하는 내용 구성들로 내용 생성하세요.
            - 콘텐츠를 JSON 형태로 작성하세요.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            입력 데이터:
            {input_text}
            
            섹션:
            {section_name}

            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            - 출력형식을 제외하고 다른 정보는 출력하지마세요.
            - 출력은 JSON형태로만 출력하세요.
            **출력 예시**:
            {{"h1" : "타이틀 내용",
            "h2" : (선택사항)"서브타이틀 내용",
            "h3" : 
            "본문" : "본문내용"}}
            """
                
        print(f"prompt length : {len(prompt)}")
        return await self.send_request(model, prompt)
    # async def landing_block_STD(self, model : str= "solar", input_text = "", section_name = "", section_num = ""):
        
    #     chunk = get_embedding_from_ollama(text=input_text)
        
    #     all_results = []
    #     for idx, chunk in enumerate(chunk):
    #         prompt = f"""
    #                 <|start_header_id|>system<|end_header_id|>
    #                 - AI 랜딩페이지 컨텐츠 생성 도우미
    #                 - 한글로 답변

    #                 <|eot_id|><|start_header_id|>user<|end_header_id|>
    #                 입력 데이터:
    #                 {chunk}
                    
    #                 섹션 이름:
    #                 {section_name}

    #                 <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    #                 - 입력 데이터 기반 컨텐츠 생성
    #                 - {section_name}에 해당하는 컨텐츠 생성
    #                 - 보고서 형식
    #                 - 코드 생성 금지
    #                 - 한글 작성
    #         """
    #         print(f"prompt length : {len(prompt)}")
    #         result = await self.send_request(model, prompt)
    #         all_results.append(result)
    #     return " ".join(all_results)
    
    
    
    # async def LLM_summary(self, input_text: str = "", model="solar"):
    #     chunk = get_embedding_from_ollama(text=input_text)
        
    #     all_results = []
    #     for idx, chunk in enumerate(chunk):
    #         prompt = f"""
    #                 <|start_header_id|>system<|end_header_id|>
    #                 당신은 고급 텍스트 요약 전문 AI 어시스턴트입니다. 다음 핵심 원칙을 엄격히 준수하세요:

    #                 요약 목표:
    #                 - 원본 텍스트의 핵심 메시지와 본질적 의미 정확하게 포착
    #                 - 불필요한 세부사항은 제외하고 핵심 내용만 추출
    #                 - 간결하고 명확한 언어로 요약
    #                 - 원문의 맥락과 뉘앙스 최대한 보존

    #                 요약 가이드라인:
    #                 - 입력된 텍스트의 주요 아이디어 식별
    #                 - 중요한 논점과 결론 강조
    #                 - 원문의 길이에 비례하여 적절한 길이로 요약
    #                 - 불필요한 반복이나 부수적인 정보 제거

    #                 요약 기법:
    #                 - 핵심 문장 추출 및 재구성
    #                 - 중요한 키워드와 주제 포함
    #                 - 논리적이고 일관된 흐름 유지

    #                 <|eot_id|><|start_header_id|>user<|end_header_id|>
    #                 입력 데이터:
    #                 {chunk}

    #                 <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    #                 요약 작성 시 다음 세부 지침 준수:

    #                 1. 입력 데이터의 본질적 의미 정확히 파악
    #                 2. 원문의 핵심 메시지를 20-30% 길이로 압축
    #                 3. 명확하고 간결한 문장 구조 사용
    #                 4. 정보의 손실 최소화
    #                 5. 읽기 쉽고 이해하기 쉬운 요약문 작성

    #                 주의사항:
    #                 - 개인적 해석이나 추가 의견 배제
    #                 - 원문의 사실관계 왜곡 금지
    #                 - 중요한 맥락이나 뉘앙스 보존
    #                 - 문법적 정확성과 가독성 확보

    #                 출력 형식:
    #                 - 명확한 주제 또는 제목
    #                 - 간결한 단락 구조
    #                 - 핵심 포인트 나열
    #                 - 논리적 흐름 유지
    #         """
            
    #         print(f"LLM_summary Len :  {len(prompt)}")
    #     return await self.send_request(model, prompt)
    
    async def LLM_content_fill(self, input_text: str = "", model="solar", summary = ""):
        
        prompt = f"""
                <|start_header_id|>system<|end_header_id|>
                당신은 전문적이고 매력적인 랜딩페이지 컨텐츠를 생성하는 고급 AI 어시스턴트입니다. 다음 지침을 철저히 따르세요:

                **주요 목표:**
                - 제공된 입력 데이터와 요약 데이터를 기반으로 컨텐츠를 작성하세요.
                - 작성된 컨텐츠는 타겟 고객의 관심을 끌 수 있도록 매력적이어야 합니다.

                **작성 지침:**
                - 모든 응답은 반드시 한글로 작성하세요.
                - 각 섹션의 형식을 유지하며 내용을 작성하세요.

                <|eot_id|><|start_header_id|>user<|end_header_id|>
                입력 데이터:
                {input_text}

                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                - 요약 데이터를 바탕으로 입력 데이터에서 필요한 내용을 도출하여 작성합니다.
                - 아래와 같은 형식으로 컨텐츠를 구성합니다:

                1. 입력 데이터의 모든 중요 정보 포함
                2. 최종 컨텐츠는 명확하고, 설득력 있으며, 전문성을 갖추도록 작성

                주의사항:
                - 문법적 오류와 부자연스러운 표현 주의
        """
        print(f"LLM_content_fill Len :  {len(prompt)}")
        return await self.send_request(model, prompt)
    
    
    async def LLM_land_page_content_Gen(self):
        """
        랜딩 페이지 섹션을 생성하고 JSON 구조로 반환합니다.
        """
        # 섹션 리스트
        section_options = ["Introduce", "Solution", "Features", "Social", 
                        "CTA", "Pricing", "About Us", "Team","blog"]

        # 섹션 수 결정 (6 ~ 9개)
        section_cnt = random.randint(6, 9)
        print(f"Selected section count: {section_cnt}")

        # 1번과 2번 섹션은 고정
        section_dict = {
            1: "Header",
            2: "Hero"
        }

        # 마지막 섹션은 Footer로 고정
        section_dict[section_cnt] = "Footer"

        # 마지막 이전 섹션에 FAQ, Map, Youtube 중 하나 배정
        minus_one_sections = ["FAQ", "Map", "Youtube", "Contact", "Support"]
        section_dict[section_cnt - 1] = random.choice(minus_one_sections)

        # 나머지 섹션을 랜덤하게 채움
        filled_indices = {1, 2, section_cnt - 1, section_cnt}
        for i in range(3, section_cnt):
            if i not in filled_indices:
                section_dict[i] = random.choice(section_options)

        # 섹션 번호 순서대로 정렬
        sorted_section_dict = dict(sorted(section_dict.items()))

        # JSON 문자열 반환
        result_json = json.dumps(sorted_section_dict, indent=4)
        print("Generated Landing Page Structure:")
        print(result_json)
        return result_json
    
    # async def LLM_land_page_content_Gen(self, input_text: str = "", model="solar", structure_limit=True):
    #     cnt = random.randint(6, 9)
    #     print(f"section count: {cnt}")
    #     try:
    #         # 비동기 함수 호출
    #         section_data = await self.landing_page_STD(model=model, input_text=input_text, section_cnt=cnt)
    #         print(f"Raw section_data: {section_data}")

    #         # JSON 형식인지 검증
    #         json_match = re.search(r"\{.*\}", section_data, re.DOTALL)
    #         json_str = json_match.group(0)
    #         # JSON 문자열을 Python 딕셔너리로 변환
    #         section_dict = json.loads(json_str)
    #         print(f"type: {type(section_dict)},\n section_dict: {section_dict}")

    #         if structure_limit:
    #             # 키를 정수로 변환
    #             section_dict = {int(k): v for k, v in section_dict.items()}

    #             # 정렬된 키 리스트
    #             keys = sorted(section_dict.keys())
    #             footer_key = keys[-1]          # 마지막 키
    #             previous_key = keys[-2]        # 마지막 키 바로 이전

    #             # 1번 섹션 검증 및 수정
    #             if section_dict[1] != "Header":
    #                 section_dict[1] = "Header"

    #             # 2번 섹션 검증 및 수정
    #             if section_dict[2] != "Hero":
    #                 section_dict[2] = "Hero"

    #             # Footer 이전 섹션 검증 및 수정
    #             minus_one = ["FAQ", "Map", "Youtube"]
    #             minus_one_num = random.randint(0, 2)

    #             if section_dict[previous_key] not in minus_one:
    #                 section_dict[previous_key] = minus_one[minus_one_num]

    #             # Footer 섹션 검증 및 수정
    #             if section_dict[footer_key] != "Footer":
    #                 section_dict[footer_key] = "Footer"

    #         print(f"type: {type(section_dict)}, len section data: {len(section_dict)}")

    #         # 최종 수정된 딕셔너리를 JSON 문자열로 변환하여 반환
    #         return json.dumps(section_dict)
    #     except json.JSONDecodeError as e:
    #         print(f"JSON decoding error: {e}")
    #         print(f"Raw response: {section_data}")
            
    #         raise HTTPException(status_code=500, detail="Invalid JSON response from LLM")
    #     except Exception as e:
    #         print(f"Error generating landing page sections: {e}")
    #         raise HTTPException(status_code=500, detail="Failed to generate landing page sections.")


    async def LLM_land_block_content_Gen(self, input_text : str = "", model = "solar", section_name = "", section_num = "1", summary=""):
        
        try:
            # 비동기 함수 호출 시 await 사용
            contents_data = await self.landing_block_STD(model=model, input_text=input_text, section_name = section_name, section_num = section_num)
            print(f"contents_data summary before: {contents_data}")
            
            # 최종 수정된 딕셔너리를 JSON 문자열로 변환하여 반환
            contents_data = await self.LLM_content_fill(model=model, input_text=contents_data, summary=summary)
            print(f"contents_data summary after: {contents_data}")
            
            return contents_data
        except Exception as e:
            print(f"Error generating landing page sections: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate landing page sections.")