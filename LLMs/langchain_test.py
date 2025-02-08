#from langchain_ollama import OllamaLLM

#llm = OllamaLLM(model="llama3.1:8b")

#llm.invoke("What is langchain?")

from langchain_community.llms import Ollama

llm = Ollama(model = "llama2")

llm.invoke("What is an LLM?")