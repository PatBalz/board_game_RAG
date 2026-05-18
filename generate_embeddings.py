import os
import pprint
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_document_DIR = os.path.join(os.getcwd(),'board_game_rules')


file_path = os.path.join(pdf_document_DIR,os.listdir(pdf_document_DIR)[0])
loader = PyPDFLoader(file_path)

docs = loader.load()
pprint.pp(len(docs))
pprint.pp(docs[0])







text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,)
texts = text_splitter.split_text(docs)
