import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    return embedding_function

def load_vector_db():
    vector_db = Chroma(
        collection_name="board_game_rules_nomic-embed-text_len1200_overlap400",
        embedding_function=get_embedding_function(),
        persist_directory="./chromaDB", 
    )
    return vector_db

def compare_splits_to_DB_content(splits_with_ids, vector_db):
    existing_items = vector_db.get(include=[])
    existing_ids = existing_items.get("ids")
    print("Current number of items in DB: "+str(len(existing_ids)))
    #only add split-items to database which are not already in database
    new_splits = []
    for split in splits_with_ids:
        if split.metadata.get("id") not in existing_ids:
            new_splits.append(split)

    split_ids_for_vectorDB = [
        new_splits[i].metadata.get("id") for i in range(len(new_splits))
        ]
    print("Number of items to add: "+str(len(new_splits)))
    return new_splits, split_ids_for_vectorDB