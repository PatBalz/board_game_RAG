import os
from langchain_community.document_loaders import PyPDFLoader
import sys

#current_dir = os.getcwd()
#src_folder = os.path.join(current_dir, "src")
#print(sys.path)
#print(src_folder.)

from knowledge_base import load_vector_db, compare_splits_to_DB_content
from splitting import split_langchain_document, get_split_ids

#PATH to Knowledge-Base
pdf_document_DIR = os.path.join(os.getcwd(),'board_game_rules')

#Create chunks and embeddinges for each document
for pdf_file in os.listdir(pdf_document_DIR):
    print("current-file: ", pdf_file)
    file_path = os.path.join(pdf_document_DIR, pdf_file)
    loader = PyPDFLoader(file_path)
    docs = loader.load() #genertes a list of langchain-documents per pdf-page

    all_splits = split_langchain_document(docs)
    print(len(all_splits))
    #bevor man die items zur DB hinzufügt müssen noch eindeitige IDs erzeugt werden
    splits_with_ids = get_split_ids(all_splits)
    #initialisiert die verbindung zur vector datenbank
    #definiert dabei die embedding function, die on-the-fly verwendet wird
    #um die embedding vectoren zu erzeugen
    vector_db = load_vector_db()
    
    new_splits, split_ids_for_vectorDB = compare_splits_to_DB_content(splits_with_ids, vector_db) 
        
    for split_idx in range(len(new_splits)):
        print(str(split_idx) + " of " + str(len(new_splits)))
        vector_db.add_documents([new_splits[split_idx]], 
                                ids = [split_ids_for_vectorDB[split_idx]])
        print("added to DB")
        print("#####")
    
#man über gibt die splits-Objecte von "split_documents" direct an die chroma_instance -> embeddings werden dann on the fly erzeugt

