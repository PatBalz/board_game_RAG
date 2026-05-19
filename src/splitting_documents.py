from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_langchain_document(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=400, 
        #add_start_index=True,
        #length_function=len
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

