from langchain_text_splitters import RecursiveCharacterTextSplitter
from semantic_chunker_langchain.chunker import SemanticChunker, SimpleSemanticChunker

def split_langchain_document(docs, split_type="recursive"):
    if split_type == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, 
            chunk_overlap=128, 
            length_function=len,
            separators=["\n\n", "\n", " ", "  ", ".", ",", "\t"]
        )
        splits = text_splitter.split_documents(docs)   

    elif split_type == "semantic":
        text_splitter = SemanticChunker(model_name="gpt-3.5-turbo")
        pre_splits = text_splitter.split_documents(docs)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, 
            chunk_overlap=20, 
            length_function=len,
        )
        splits = text_splitter.split_documents(pre_splits)
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

