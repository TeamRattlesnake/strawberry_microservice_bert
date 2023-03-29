import os
import re

from tqdm import tqdm
import logging

import torch
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

from transformers import T5ForConditionalGeneration, T5Tokenizer

logging.basicConfig(format="%(asctime)s %(message)s", handlers=[logging.FileHandler(
    f"/home/logs/summarize_log_model.txt", mode="w", encoding="UTF-8")], datefmt="%I:%M:%S %p", level=logging.INFO)


class NeuralNetwork:
    def __init__(self, group_id=0):
        self.DEVICE = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = "cointegrated/rut5-base-absum"
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        self.group_id = group_id

    def generate(self, hint, n_words=None, compression=None, max_length=1000, num_beams=3, do_sample=False, repetition_penalty=10.0, **kwargs):
        logging.info(f"generating (summarizing) for {hint}")
        if n_words:
            hint = f"[{n_words}] {hint}"
        elif compression:
            hint = f"[{compression}] {hint}"
        x = self.tokenizer(hint, return_tensors='pt',
                           padding=True).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **x,
                max_length=max_length, num_beams=num_beams,
                do_sample=do_sample, repetition_penalty=repetition_penalty,
                **kwargs
            )
        generated_text = self.tokenizer.decode(
            out[0], skip_special_tokens=True)
        logging.info(f"generating (summarizing) for {hint}: {generated_text}")
        return generated_text
