from transcript_loader import aquire_transcript
import os
import yaml
import torch
from torch import package
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                               'latest_silero_models.yml', progress=False)

with open('latest_silero_models.yml', 'r', encoding='utf8') as yaml_file:
    models = yaml.load(yaml_file, Loader=yaml.SafeLoader)
model_conf = models.get('te_models').get('latest')
model_url = model_conf.get('package')

model_dir = "downloaded_model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, os.path.basename(model_url))

if not os.path.isfile(model_path):
    torch.hub.download_url_to_file(model_url, model_path, progress=True)

def punctuate(text, lan='en'):
    imp = package.PackageImporter(model_path)
    model = imp.load_pickle("te_model", "model")
    return model.enhance_text(text, lan)

def split_text(documents, chunk_size=1000, chunk_overlaps=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlaps)
  docs = text_splitter.split_text(documents)
  return docs

if __name__ == "__main__":
    link = input("Enter the link: ")
    text = aquire_transcript(link)
    text = punctuate(text)
    print(text)
