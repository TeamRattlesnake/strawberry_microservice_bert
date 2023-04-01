import os
import re
import string

from tqdm import tqdm
import logging

import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

from transformers import T5ForConditionalGeneration, T5Tokenizer

logging.basicConfig(format="%(asctime)s %(message)s", handlers=[logging.FileHandler(
    f"/home/logs/bert_log_model.txt", mode="w", encoding="UTF-8")], datefmt="%I:%M:%S %p", level=logging.INFO)


class NeuralNetwork:
    def __init__(self, group_id=0):
        self.pipe = pipeline(model="sberbank-ai/ruBert-base")

    def generate(self, hint):
        logging.info(f"Generating for hint: {hint}")
        result = self.pipe(hint, top_k=10)

        result = [item["token_str"] for item in result]

        result = [item for item in result if re.match(
            "^[A-Za-zа-яА-ЯёЁ]*$", item)]

        result = ",".join(result)
        logging.info(f"Generated for hint: {hint}: {result}")
        return result
