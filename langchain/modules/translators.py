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
    
    def translate_length_limit(self, text: str, max_token_length: int = 64, max_total_tokens: int = 6000) -> str:
        """
        긴 텍스트를 분할하여 번역하고, 총 번역된 토큰 수가 max_total_tokens를 초과하면 중단합니다.
        
        Args:
            text (str): 번역할 긴 텍스트.
            max_token_length (int): 각 청크의 최대 토큰 수.
            max_total_tokens (int): 번역된 전체 텍스트의 최대 토큰 수.
        
        Returns:
            str: 번역된 텍스트의 합계 (최대 max_total_tokens 토큰).
        """
        if not isinstance(text, str):
            raise ValueError(f"translate 메소드는 문자열을 기대합니다. 전달된 타입: {type(text)}")
        
        # 텍스트를 청크로 분할
        chunks = self.split_text(text, max_token_length)

        translated_chunks = []
        total_tokens = 0

        for chunk in chunks:
            # 청크를 토크나이즈하여 모델 입력 준비
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_token_length
            ).to(self.device)

            # 모델을 사용하여 번역
            outputs = self.model.generate(
                **inputs,
                max_length=max_token_length,
                num_beams=5,
                early_stopping=True
            )

            # 번역된 청크 디코딩
            translated_chunk = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 번역된 청크의 토큰 수 계산
            translated_chunk_tokens = len(self.tokenizer.encode(translated_chunk))

            # 누적 토큰 수가 최대 한계를 초과하지 않도록 체크
            if total_tokens + translated_chunk_tokens > max_total_tokens:
                remaining_tokens = max_total_tokens - total_tokens
                if remaining_tokens > 0:
                    # 남은 토큰 수에 맞게 청크를 트렁케이트
                    truncated_tokens = self.tokenizer.encode(translated_chunk)[:remaining_tokens]
                    truncated_chunk = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    translated_chunks.append(truncated_chunk)
                    total_tokens += len(truncated_tokens)
                break  # 최대 토큰 수에 도달했으므로 루프 종료
            else:
                translated_chunks.append(translated_chunk)
                total_tokens += translated_chunk_tokens

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
