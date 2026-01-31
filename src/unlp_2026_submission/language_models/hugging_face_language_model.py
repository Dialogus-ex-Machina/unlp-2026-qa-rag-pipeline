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
                max_new_tokens=config.language_model_max_tokens,
                temperature=config.language_model_temperature,
                top_k=config.language_model_top_k,
                top_p=config.language_model_top_p,
                repetition_penalty=config.language_model_repeat_penalty,
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
                max_new_tokens = config.language_model_max_tokens,
                temperature = config.language_model_temperature,
                top_k = config.language_model_top_k,
                top_p = config.language_model_top_p,
                repetition_penalty = config.language_model_repeat_penalty,
                do_sample = True,
                return_full_text = False,
            ),
        )
