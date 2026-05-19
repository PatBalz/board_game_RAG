from src.knowledge_base import load_vector_db, get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import pprint

vector_db = load_vector_db()
embedding_function = get_embedding_function()

PROMPT_TEMPLATE = """
Please answer the given question only based on the context below:
{context}
Please answer the given question only based on the context above: {question}
Please be honest if you are not confident in your answer. In this 
case try to give hints to the user on where to search for an answer to
the question. 
"""


question = "when does the game end?"

search_results = vector_db.similarity_search_with_score(question, k = 5)

for res in search_results:
    print("Result, Distance =  "+str(res[1]))
    print(res[0].metadata.get("source"))
    print(" ".join(res[0].page_content.split("\\n")))
    print("\n ------ \n")
 
context_text = "\n --- \n".join(["Source: "+res[0].metadata.get("source")+"\n Content: "+res[0].page_content for res in search_results])

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=question)

model = OllamaLLM(
    model="llama3.2:1b",
    #temperature=0.4,
    #num_ctx=1250,
    #num_predict=4000
    # base_url="http://localhost:11434",
    # other params...
)

for chunk in model.stream(prompt):
    print(chunk, end = "")


