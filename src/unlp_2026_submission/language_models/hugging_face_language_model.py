from __future__ import annotations

import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from unlp_2026_submission.config import Config

class HuggingFaceLanguageModel(ChatHuggingFace):
    @staticmethod
    def create(config: Config) -> ChatHuggingFace:
        llm = HuggingFacePipeline.from_model_id(
            model_id=config.language_model_name,
            task="text-generation",
            pipeline_kwargs=dict(
                max_new_tokens=2048,
                temperature=0.1,
                top_k=25,
                top_p=1,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False,
            ),
            model_kwargs=dict(
                dtype=torch.bfloat16,
                cache_dir=config.downloaded_models_cache_dir,
                device_map="auto",
            )
        )

        return ChatHuggingFace(llm=llm)


class LlamaIndexHuggingFaceLanguageModel(HuggingFaceLLM):
    @staticmethod
    def create(config: Config) -> HuggingFaceLLM:
        model = AutoModelForCausalLM.from_pretrained(
            config.language_model_name,
            dtype=torch.bfloat16,
            cache_dir=config.downloaded_models_cache_dir,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.language_model_name,
            cache_dir=config.downloaded_models_cache_dir,
            use_default_system_prompt=False,
        )

        return HuggingFaceLLM(
            model_name=config.language_model_name,
            model=model,
            tokenizer=tokenizer,
            generate_kwargs=dict(
                max_new_tokens=2048,  # Choose maximum generation tokens
                temperature=0.1,
                top_k=25,
                top_p=1,
                repetition_penalty=1.1,
                do_sample=True
            ),
        )
