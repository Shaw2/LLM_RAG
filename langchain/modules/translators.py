from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from models.models_conf import KO_EN_MODEL_PATH, EN_KO_MODEL_PATH
from config.config import DEVICE, MAX_LENGTH
import torch


class KoEnTranslator:
    """
    한국어 -> 영어 번역기 클래스
    """
    def __init__(self):
        self.device = torch.device(DEVICE)
        print(f"Initializing KoEnTranslator on device: {self.device}")

        # 한국어 -> 영어 모델과 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(KO_EN_MODEL_PATH)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(KO_EN_MODEL_PATH).to(self.device)
        self.max_length = MAX_LENGTH

    def split_text(self, text, max_length):
        """
        긴 텍스트를 max_length 기준으로 슬라이싱
        """
        tokens = text.split()
        chunks = []
        current_chunk = []

        for token in tokens:
            if len(" ".join(current_chunk + [token])) > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [token]
            else:
                current_chunk.append(token)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks


    def translate(self, text: str, max_token_length: int = 64) -> str:
        """
        긴 텍스트를 분할하여 번역
        """
        print("한국어 번역 시작")

        # 긴 텍스트 분할
        chunks = self.split_text(text, max_token_length)

        # 각 조각을 번역하고 결과를 합치기
        translated_chunks = []
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_length=max_token_length,
                num_beams=5,
                early_stopping=True
            )

            translated_chunks.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

        return " ".join(translated_chunks)

    # def translate(self, text: str, max_token_length: int = 1024) -> str:
    #     """
    #     한국어 텍스트를 영어로 번역합니다.
    #     """
    #     print("한국어 번역 시작")
    #     # 입력 텍스트를 토큰화
    #     inputs = self.tokenizer(
    #         text, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length
    #     ).to(self.device)

    #     # 모델로 번역 생성
    #     outputs = self.model.generate(
    #         **inputs,
    #         max_length=max_token_length,
    #         num_beams=5,
    #         early_stopping=True
    #     )

    #     # 번역 결과 디코딩
    #     translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return translated_text

    def translate_batch(self, texts: list[str], max_token_length: int = 512) -> list[str]:
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_token_length,
            num_beams=5,
            early_stopping=True
        )
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


class EnKoTranslator:
    """
    영어 -> 한국어 번역기 클래스
    """
    def __init__(self):
        self.device = torch.device(DEVICE)
        print(f"Initializing EnKoTranslator on device: {self.device}")

        # 영어 -> 한국어 모델과 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(EN_KO_MODEL_PATH)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(EN_KO_MODEL_PATH).to(self.device)

    def split_text(self, text, max_length):
        """
        긴 텍스트를 max_length 기준으로 슬라이싱
        """
        tokens = text.split()
        chunks = []
        current_chunk = []

        for token in tokens:
            if len(" ".join(current_chunk + [token])) > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [token]
            else:
                current_chunk.append(token)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks


    def translate(self, text: str, max_token_length: int = 64) -> str:
        """
        긴 텍스트를 분할하여 번역
        """

        # 긴 텍스트 분할
        if not isinstance(text, str):
            raise ValueError(f"translate 메소드는 문자열을 기대합니다. 전달된 타입: {type(text)}")
        
        chunks = self.split_text(text, max_token_length)

        # 각 조각을 번역하고 결과를 합치기
        translated_chunks = []
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_length=max_token_length,
                num_beams=5,
                early_stopping=True
            )

            translated_chunks.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

        return " ".join(translated_chunks)
    
    def translate_structure(self, data):
        if isinstance(data, dict):
            return {k: self.translate_structure(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.translate_structure(item) for item in data]
        elif isinstance(data, str):
            return self.en_ko_translator.translate(data)
        else:
            return data
        
    # def translate(self, text: str, max_token_length: int = 1024) -> str:
    #     """
    #     영어 텍스트를 한국어로 번역합니다.
    #     """
    #     # 입력 텍스트를 토큰화
    #     inputs = self.tokenizer(
    #         text, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length
    #     ).to(self.device)

    #     # 모델로 번역 생성
    #     outputs = self.model.generate(
    #         **inputs,
    #         max_length=max_token_length,
    #         num_beams=5,
    #         early_stopping=True
    #     )

    #     # 번역 결과 디코딩
    #     translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return translated_text

    def translate_batch(self, texts: list[str], max_token_length: int = 1024) -> list[str]:
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_token_length,
            num_beams=5,
            early_stopping=True
        )
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
