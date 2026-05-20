from src.knowledge_base import load_vector_db, get_embedding_function, perform_vector_search
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

question = "how many intrigue cards are part of the game?"#"what do I do during an agent turn?"#"on which board spaces can I gain spice?"#"how to gain victory points?"#"what can I do with a combat intrigue card?"#"what are the different types of intrigue cards?"#"what are intrigue cards, what can I do with them?"#"when does the endgame start?"#"""what happens when I send an Agent to Conspire board space and what do I gain?"""#"how do I get the Mentat"
#"how many victory points does one player need to reach in order to win the game?"
#"how do I get the Mentat"# "on which board spaces can I gain spice?"
 
context_text = perform_vector_search(vector_db, question)
print(context_text)

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=question)

model = OllamaLLM(
    model="llama3.2:1b",
    temperature=0.0,
    )

print("--------------------------")
print("Question was: \n"+question)
for chunk in model.stream(prompt):
    print(chunk, end = "")



#baue hier eine reasoning schleife ein mit der das llm selber nochmal eine frage an die datenbank stellen kann

###
##TODO
#Text muss sinnvoller gesplittet werden, vielleicht erst aus pdf rauskopieren? (done)
#logische einheiten hängen gerade nicht korrekt zusammen
#vielleicht muss ich eine eigene funktion schreiben, die eine semantische trennungs vornimmt (sliding window ansatz, dann embeddings vergleichen)
#vielleicht macht es sinn nochmal nach besseren splittern für das original pdf zu schauen, oder auch hier einen selber zu schreiben
