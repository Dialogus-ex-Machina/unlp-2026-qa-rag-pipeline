import torch
from langchain.chat_models import init_chat_model
from transformers import BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32,
)


def _create_hf_chat_model(
    model_id: str,
    *,
    do_sample: bool = False,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    min_new_tokens: int = 16,
    attn_implementation: str = "sdpa",
):
    pipeline_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "return_full_text": False,
        "renormalize_logits": True,
        "remove_invalid_values": True,
        "truncation": True,
    }
    if do_sample:
        pipeline_kwargs["temperature"] = temperature

    return init_chat_model(
        model_id,
        model_provider="huggingface",
        model_kwargs={
            "quantization_config": bnb,
            "device_map": "cuda",
            "dtype": torch.float16,
            "trust_remote_code": True,
            "attn_implementation": attn_implementation,
        },
        pipeline_kwargs=pipeline_kwargs,
    )


class HuggingFaceLanguageModel:
    @staticmethod
    def create():
        return HuggingFaceLanguageModel.create_mamaylm()

    @staticmethod
    def create_mamaylm():
        return _create_hf_chat_model(
            "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0",
            do_sample=False,
            max_new_tokens=512,
        )

    @staticmethod
    def create_qwen2_5_3b():
        return _create_hf_chat_model(
            "Qwen/Qwen2.5-3B-Instruct",
            do_sample=False,
            max_new_tokens=512,
        )

    @staticmethod
    def create_gemma2_2b_it():
        return _create_hf_chat_model(
            "google/gemma-2-2b-it",
            do_sample=False,
            max_new_tokens=512,
        )

    @staticmethod
    def create_phi3_mini_4k():
        return _create_hf_chat_model(
            "microsoft/Phi-3-mini-4k-instruct",
            do_sample=False,
            max_new_tokens=512,
            attn_implementation="sdpa",
        )

    @staticmethod
    def create_other(
        model_id: str,
        *,
        do_sample: bool = False,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        min_new_tokens: int = 16,
        attn_implementation: str = "sdpa",
    ):
        return _create_hf_chat_model(
            model_id,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            attn_implementation=attn_implementation,
        )
