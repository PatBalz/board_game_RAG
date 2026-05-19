from src.knowledge_base import load_vector_db, get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import pprint

vector_db = load_vector_db()
embedding_function = get_embedding_function()

PROMPT_TEMPLATE = """
You are an instructor for people who participate in a game night. 
They play the board game dune imperium and potentially some addons to it.
Players may have questions on certain rules or cards, because the rulebook of this game is very
extensive and hard to understand. You are there last hope to really understand the game rules.
Please treat any question as a completely new interaction without remembering what was done before.
Please answer the given question in english based on the context below:
{context}
Here is the question:
{question}
Remember to stick to the context above to answer the question!
Please be honest if you are not confident in your answer.
"""


question = "one player hase reached 10 victory points, what to do now?"


search_results = vector_db.similarity_search_with_score(question, k = 20)


for res in search_results:
    print("Result, Distance =  "+str(res[1]))
    #print(res[0].metadata.get("source"))
    #print(res[0].page_content.split("\\n"))
    #print("\n ------ \n")
 
context_text = "\n --- \n".join(["Source: "+res[0].metadata.get("source")+"\n Content: "+res[0].page_content for res in search_results])
print(context_text)
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=question)

model = OllamaLLM(
    model="llama3.2:1b",
    temperature=0.5,
    #num_ctx=500,
    #num_predict=4000
    # base_url="http://localhost:11434",
    # other params...
)
for res in search_results:
    print("Result, Distance =  "+str(res[1]))
print("Your question was: "+question)
print("-------")
for chunk in model.stream(prompt):
    print(chunk, end = "")



#baue hier eine reasoning schleife ein mit der das llm selber nochmal eine frage an die datenbank stellen kann

###
#du musst dringend die embeddings anpassen, diese sind der letzte dreck
#danach bitte einmal explorieren, wie man die qualität der modelle testen kann