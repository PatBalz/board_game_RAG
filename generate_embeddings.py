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



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))

from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3")



from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="board_game_rules",
    embedding_function=embeddings,
    persist_directory="./chromaDB", 
)