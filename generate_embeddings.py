import os
import pprint
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

###Helper Functions
def get_embedding_function():
    embedding_function = OllamaEmbeddings(model="llama3")
    return embedding_function

def split_langchain_document(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    return splits

def get_split_ids(splits):
    last_page_number = None
    split_idx = 0
    for split in splits:
        source_file = split.metadata.get("title")
        page_number = split.metadata.get("page")
        if page_number == last_page_number:
            split_idx = split_idx + 1
        else:
            split_idx = 0
        current_id = f"{source_file}:{page_number}:{split_idx}"
        split.metadata["id"] = current_id
        last_page_number = page_number
    return splits


pdf_document_DIR = os.path.join(os.getcwd(),'board_game_rules')

#get one file
for pdf_file in os.listdir(pdf_document_DIR)[0:1]:
    print("current-file: ", pdf_file)
    file_path = os.path.join(pdf_document_DIR, pdf_file)
    loader = PyPDFLoader(file_path)
    docs = loader.load() #genertes a list of langchain-documents per pdf-page
    #pprint.pp(len(doc))
    #pprint.pp(doc[0]) 

    all_splits = split_langchain_document(docs)
    print(len(all_splits))
  
    #bevor mandie items zur DB hinzufügt müssen noch eindeitige IDs erzeugt werden
    splits_with_ids = get_split_ids(all_splits)
    
    #initialisiert die verbindung zur vector datenbank
    #definiert dabei die embedding function, die on-the-fly verwendet wird
    #um die embedding vectoren zu erzeugen
    #ollama server muss dafür laufen
    vector_db = Chroma(
        collection_name="board_game_rules",
        embedding_function=get_embedding_function(), #hier wird eine embedding function aufgerufen, aber wie bekomme ich da den text rein?
        persist_directory="./chromaDB", 
    )

    #get existing items in vector database
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
    
    for split_idx in range(len(new_splits)):
        print(str(split_idx) + " of " + str(len(new_splits)))
        vector_db.add_documents([new_splits[split_idx]], 
                                ids = [split_ids_for_vectorDB[split_idx]])
        print("added to DB")
        print("#####")
    vector_db.persist()
#man über gibt die splits-Objecte von "split_documents" direct an die chroma_instance -> embeddings werden dann on the fly erzeugt


