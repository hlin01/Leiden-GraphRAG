# main.py
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

load_dotenv()
os.environ["OPENAI_API_KEY"]

with open("anonymized_data_cl1.txt", "r") as f:
    notes = f.readlines()

documents = [
    Document(text=note.strip() for note in notes)
]

