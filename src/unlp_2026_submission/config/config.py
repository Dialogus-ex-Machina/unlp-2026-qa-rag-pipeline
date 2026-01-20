import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    language_model_name: str

    def __init__(self):
        self.language_model_name = os.getenv('LANGUAGE_MODEL_NAME')
