from src.knowledge_base import load_vector_db, get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import pprint

vector_db = load_vector_db()
embedding_function = get_embedding_function()

PROMPT_TEMPLATE = """
Please answer this question : '{question}' only by using the context given below:
{context}
"""
"""
Setting: 
You are an instructor for people who participate in a game night.
These peolple are a group of friends which enjoy complex board games. 
Players may have questions on certain rules or cards, because the rulebook of this game is very
extensive and hard to understand. You are there last hope to really understand the game rules.
Further below you will find chunks of information directly drawn from the rule book. Please try to logically deduce 
the answer to the following question, based on the fragments and instructions given below!
This is our question: {question}
Plsease summarize the question in your own terms! Then, please answer the given question based on these following context fragments from the rule book below:
{context}
Remember to stick to the context above to answer the question! But keep in mind that maybe not all fragments 
are helpful to answer the question. Most of the fragments will mostlikely not be helpful for you! Therefore, please use the following structure for your internal reasoning:
1. be aware that fragments can contain only partially useful information with regard to the question or 
that the useful information is hidden within a fragment, please pay only attention to the parts of the fragment which are relevant to 
answer the question: {question}
2. write a summary for each fragment and estimate the relevance of the fragment for how much
 it helps answering the question '{question}'. But only consider the fragment if it is related to the words within the question: '{question}'
3. answer the question only based on the important fragments and in terms of relevant game actions
Please keep your answers as concicse as possible and do not repeat yourself! 
Please use the following response structure:
[Fragment and Relevanve Score]
[Conclusion towards answering the question]
Finally, take your final answer and analyse if the words used in your answer have any connection to the original question '{question}'. 
"""


question = "how do I get the Mentat"##"""what happens when I send an Agent to Conspire board space and what do I gain?"""
#"how many victory points does one player need to reach in order to win the game?"
#"how do I get the Mentat"#


search_results = vector_db.similarity_search_with_score(question, k = 5)


for res in search_results:
    print("Result, Distance =  "+str(res[1]))
    #print(res[0].metadata.get("source"))
    #print(res[0].page_content.split("\\n"))
    #print("\n ------ \n")
 
context_text = "\n --- \n".join(["\n Source: "+res[0].metadata.get("id")+"\n Content from Rulebook: [START OF CONTENT FRAGMENT] \n"+res[0].page_content+" [END OF CONTENT FRAGMENT]\n" for res in search_results])
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
#Text muss sinnvoller gesplittet werden, vielleicht erst aus pdf rauskopieren? 
#logische einheiten hängen gerade nicht korrekt zusammen