from src.knowledge_base import load_vector_db, get_embedding_function
from src.misc import load_prompt_template, load_config
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import yaml
import os

BASE_PATH = os.getcwd()
CONFIG_PATH = os.path.join(BASE_PATH, "config\config.yml")
config = load_config(CONFIG_PATH)
PROMPT_TEMPLATE_PATH = os.path.join(BASE_PATH, config["prompt_template"])


vector_db = load_vector_db()
embedding_function = get_embedding_function()

PROMPT_TEMPLATE = load_prompt_template(PROMPT_TEMPLATE_PATH)

question = "how to gain victory points?"#"what can I do with a combat intrigue card?"#"what are the different types of intrigue cards?"#"what are intrigue cards, what can I do with them?"#"when does the endgame start?"#"""what happens when I send an Agent to Conspire board space and what do I gain?"""#"how do I get the Mentat"
#"how many victory points does one player need to reach in order to win the game?"
#"how do I get the Mentat"# "on which board spaces can I gain spice?"

search_results = vector_db.similarity_search_with_score(question, k = 5)

for res in search_results:
    print("Result, Distance =  "+str(res[1]))
    #print(res[0].metadata.get("source"))
    #print(res[0].page_content.split("\\n"))
    #print("\n ------ \n")
 
context_text = "\n --- \n".join(["\n Source: "+res[0].metadata.get("id")+"\n Content from Rulebook: [START OF CONTENT FRAGMENT] \n"+res[0].page_content+" [END OF CONTENT FRAGMENT]\n" for res in search_results if res[1]<=10])
print(context_text)
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=question)

model = OllamaLLM(
    model="llama3.2:1b",
    temperature=0.0,
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
##TODO
#Text muss sinnvoller gesplittet werden, vielleicht erst aus pdf rauskopieren? (done)
#logische einheiten hängen gerade nicht korrekt zusammen
#vielleicht muss ich eine eigene funktion schreiben, die eine semantische trennungs vornimmt (sliding window ansatz, dann embeddings vergleichen)
#vielleicht macht es sinn nochmal nach besseren splittern für das original pdf zu schauen, oder auch hier einen selber zu schreiben
