import os
from langchain_community.document_loaders import PyPDFLoader
from src.knowledge_base import load_vector_db, compare_splits_to_DB_content
from src.splitting_documents import split_langchain_document, get_split_ids
from src.misc import load_config
from tqdm import tqdm

#Configuration settings
BASE_PATH = os.getcwd()
CONFIG_PATH = os.path.join(BASE_PATH, "config\config.yml")
config = load_config(CONFIG_PATH)
KNOWLEDGE_PATH = os.path.join(BASE_PATH, config["knowledge_folder"])
SPLIT_TYPE = config["split_type"]

#Create chunks and embeddinges for each document
for pdf_file in os.listdir(KNOWLEDGE_PATH):
    print("current-file: ", pdf_file)
    file_path = os.path.join(KNOWLEDGE_PATH, pdf_file)
    loader = PyPDFLoader(file_path)
    docs = loader.load() #genertes a list of langchain-documents per pdf-page

    all_splits = split_langchain_document(docs, SPLIT_TYPE)
    print(len(all_splits))
    #bevor man die items zur DB hinzufügt müssen noch eindeitige IDs erzeugt werden
    splits_with_ids = get_split_ids(all_splits)
    #initialisiert die verbindung zur vector datenbank
    #definiert dabei die embedding function, die on-the-fly verwendet wird
    #um die embedding vectoren zu erzeugen
    vector_db = load_vector_db()
    
    new_splits, split_ids_for_vectorDB = compare_splits_to_DB_content(splits_with_ids, vector_db) 
        
    for split_idx in tqdm(range(len(new_splits))):
        #print(str(split_idx) + " of " + str(len(new_splits)))
        vector_db.add_documents([new_splits[split_idx]], 
                                ids = [split_ids_for_vectorDB[split_idx]])
        #print("added to DB")
        #print("#####")
    
#man über gibt die splits-Objecte von "split_documents" direct an die chroma_instance -> embeddings werden dann on the fly erzeugt

