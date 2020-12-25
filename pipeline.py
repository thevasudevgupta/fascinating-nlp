import torch
from transformers import pipeline
import re

def measure_time(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        print(f"total time taken is {round(time.time() - start, 2)} seconds")
        return output
    return wrapper


class Pipeline(object):

    def __init__(self, translator, summarizer):
        self.translator = pipeline("translation_hi_to_en", model=translator["model"], tokenizer=translator["tokenizer"])
        self.summarizer = pipeline("summarization", model=summarizer["model"], tokenizer=summarizer["tokenizer"])

    @measure_time
    def __call__(self, paragraph:str, min_length:int, max_length:int):

        inputs = re.split("[.?|]", paragraph)
        while "" in inputs: inputs.remove("")
        translation = self.translator(inputs, return_text=True)
        translation = [t["translation_text"] for t in translation]
        del self.translator

        inputs = ".".join(translation)
        summary = self.summarizer(inputs, return_text=True, min_length=min_length, max_length=max_length)
        del self.summarizer

        return [s["summary_text"] for s in summary], translation
