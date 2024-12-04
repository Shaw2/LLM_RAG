from langchain.prompts import PromptTemplate
from modules.translators import KoEnTranslator, EnKoTranslator
from utils.ollama_client import OllamaClient


class ContentChain:
    def __init__(self):
        self.ko_en_translator = KoEnTranslator()
        self.en_ko_translator = EnKoTranslator()
        self.ollama_client = OllamaClient()

        # self.text_generator = TextGenerator()
        # 프롬프트 템플릿 설정
        self.prompt_template = PromptTemplate(
            input_variables=["text"], 
            template="""- You are the organizer of the company introduction website. 
             - Data is a company profile or company introduction. 
             - The result values will be printed in three ways.
             - First is the title of website.
             - Second is pick 3 keywords in the contents.
             - Third is to write two_depth menu refer to menu_structure in assistant. The first_depth must be 3~5. The second_depth must be 0~4. Don't need to write extra explaination about second_depth. the length of menus should be less than 15 letters.
             
             ex) : 
             예시:
            {
                "Title": "company introduce",
                "Keywords": ["sofa", "interior", "furniture"],
                "menu": {
                    "소개": ["Company Overview", "history", "vision"],
                    "서비스": ["product descriptions", "solution", "support"],
                    "연락처": ["Contact us", "Location", "Recruit"]
                }
            }
            {text}
             """
        )
    # 일괄 처리 방식
    def run(self, input_text, discriminant, model="llama3.2"):
        """
        Ollama API 기반 텍스트 생성 체인
        Args:
            input_text (str): 입력 텍스트
            discriminant (bool): 한국어 여부
            model (str): Ollama에서 사용할 모델 이름

        Returns:
            str: 최종 생성 결과
        """
        print("run operation success")
        if discriminant:
            # 한->영 번역
            translated_text = self.ko_en_translator.translate(input_text)
            print("Translated to English:", translated_text)

            # Ollama API 호출
            generated_text = self.ollama_client.generate(model, translated_text)
            print("Generated Text:", generated_text)

            # 영->한 번역
            final_output = self.en_ko_translator.translate(generated_text)
            print("Translated back to Korean:", final_output)
        else:
            # Ollama API 호출
            generated_text = self.ollama_client.generate(model, input_text)
            print("Generated Text:", generated_text)

            # 영->한 번역
            final_output = self.en_ko_translator.translate(generated_text)
            print("Translated back to Korean:", final_output)
        
        return final_output.strip()
    # streaming 처리 방식
    # def run(self, input_text, discriminant, model="llama3.2"):
    #     """
    #     Ollama API 기반 텍스트 생성 체인
    #     Args:
    #         input_text (str): 입력 텍스트
    #         discriminant (bool): 한국어 여부
    #         model (str): Ollama에서 사용할 모델 이름

    #     Returns:
    #         str: 최종 생성 결과
    #     """
    #     print("run operation success")
    #     if discriminant:
    #         # 한->영 -> 텍스트 생성 -> 영->한
    #         translated_text = self.ko_en_translator.translate(input_text)
    #         final_output = self._stream_generate_and_translate(model, translated_text, is_korean=True)
    #     else:
    #         # 영문 텍스트 바로 생성 -> 영->한
    #         final_output = self._stream_generate_and_translate(model, input_text, is_korean=False)
        
    #     return final_output
    
    def _stream_generate_and_translate(self, model, input_text, is_korean):
        """
        스트리밍 방식으로 Ollama 텍스트 생성 및 번역 처리
        Args:
            model (str): Ollama에서 사용할 모델 이름
            input_text (str): 입력 텍스트
            is_korean (bool): 한국어 여부

        Returns:
            str: 최종 생성 결과
        """
        # 스트리밍 데이터 받아오기
        streamed_text = self.ollama_client.generate(model, input_text)
        
        # 실시간 번역 처리
        translated_output = ""
        for chunk in streamed_text.split(" "):  # 단어 단위로 스트리밍 처리
            if is_korean:
                translated_chunk = self.en_ko_translator.translate(chunk)
            else:
                translated_chunk = chunk
            translated_output += translated_chunk + " "
            print("Translated Chunk:", translated_chunk)  # 디버깅용 출력
        
        return translated_output.strip()
    
    
    # def run(self, input_text, discriminant):
    #     if discriminant:
    #         # 한->영 -> 텍스트 생성 -> 영->한
    #         translated_text = self.ko_en_translator.translate(input_text)
    #         generated_text = self.text_generator.generate(self.prompt_template.format(text=translated_text))
    #         final_output = self.en_ko_translator.translate(generated_text)
    #     else:
    #         # 영문 텍스트 바로 생성 -> 영->한
    #         generated_text = self.text_generator.generate(self.prompt_template.format(text=input_text))
    #         final_output = self.en_ko_translator.translate(generated_text)
    #     return final_output
