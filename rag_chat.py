from src.knowledge_base import load_vector_db, get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import pprint

vector_db = load_vector_db()
embedding_function = get_embedding_function()

PROMPT_TEMPLATE = """
Please answer the given question only based on the context below:
{context}
Please answer the given question only based on the context above: {question}
Only commit to a final answer only if you are really confident in it. 
Please be honest if you are not confident in your answer. In this 
case try to give hints to the user on where to search for an answer to
the question
"""


question = "How can I deploy a unit in dune imperium?"

search_results = vector_db.similarity_search_with_score(question, k = 5)

for res in search_results:
    print("Result, DIstance =  "+str(res[1]))
    pprint.pp(res[0].metadata.get("source"))
    pprint.pp(res[0].page_content)
#context_text = #muss aus den results erzeugt werden

#prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE, )
#prompt = prompt.format(context=context_text, question=question)

