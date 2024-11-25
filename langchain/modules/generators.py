# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from langchain.config import LLAMA_MODEL_PATH, DEVICE, LANGCHAIN_DEFAULT_PROMPT, MAX_LENGTH


# class TextGenerator:
#     def __init__(self):
#         self.device = torch.device(DEVICE)
#         print(f"Using device: {self.device}")

#         # 모델과 토크나이저 초기화
#         self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
#         self.model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_PATH).to(self.device)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.system_prompt = LANGCHAIN_DEFAULT_PROMPT

#     def generate(self, prompt, max_length=MAX_LENGTH):
#         full_prompt = f"{self.system_prompt} {prompt}"
#         input_ids = self.tokenizer.encode(full_prompt, 
#                                           return_tensors="pt", 
#                                           padding=True, 
#                                           truncation=True).to(self.device)
        
#         outputs = self.model.generate(input_ids,
#                                       max_length=max_length,
#                                       pad_token_id=self.tokenizer.pad_token_id)
        
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
