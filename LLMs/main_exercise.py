from openai import OpenAI
import dotenv
import os
from rich import print as rprint

dotenv.load_dotenv()

# Global parameters
MAX_NUM_TOKENS_RESPONSE = 1000

# API call
client = OpenAI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# give it prompt
system_prompt = "You are an expert on computational fluid dynamics and gas turbine and ducted fan design"
user_query = "Can you please tell me how to optimize efficiency of a ducted fan for a drone?"

response = client.chat.completions.create(
    model= "gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ],
    max_tokens=128
) 

print("Simple prompt \n")
print(response.choices[0].message.content)
print("*******************************************\n")

# give it prompt v2
# chain of thought prompting
#   https://docs.science.ai.cam.ac.uk/hands-on-llms/prompting/3_prompting/#piagets-glass-of-water
system_prompt = "You are an expert on computational fluid dynamics and gas turbine and ducted fan design"
user_query = "Can you please tell me how to optimize efficiency of a ducted fan for a drone? Assume you have access to the diffusion factor for flow through the turbine. Please step through your reasoning and explain all the steps in your reasoning. For example, start byd efining efficiency, then determine how diffusion factor will affect efficiency, then determine how to change diffusion factor. Please provide references and links to justify your arguments."

response = client.chat.completions.create(
    model= "gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ],
    max_tokens=MAX_NUM_TOKENS_RESPONSE
) 

print("*******************************************\n")
print("Chain of thought prompting \n")
print(response.choices[0].message.content)
print("\n*******************************************\n")


print("More complex prompt on automated scientific discovery \n")
print("*******************************************\n")

# prompt on automated scientific discovery
#system_prompt = "You are an expert on scientific discovery. You are tasked with finding the relationship between variables. "
#user_query = "You are given the values of variables in a tabular format. Please provide me the relationship between these variables? The values are given in a table in latex code here: \begin{tabular}{|c c c c c c c|} \hline Planet & Distance ($D$) & Period ($P$) & $\frac{D}{P}$ & $\frac{D^2}{P}$ & $\frac{D^2}{P^2}$ & $\frac{D^3}{P^2}$ \\\hline $A$ & 1.0 & 1.0 & 1.0 & 1.0 & 1.0 & 1.0 \\ $B$ & 4.0 & 8.0 & 0.5 & 2.0 & 0.25 & 1.0 \\ $C$ & 9.0 & 27.0 & 0.333 & 3.0 & 0.111 & 1.0 \\ \hline \end{tabular} "

#response = client.chat.completions.create(
#    model= "gpt-4o-mini",
#    messages=[
#        {"role":"system", "content": system_prompt},
#        {"role":"user", "content": user_query},
#    ],
#    max_tokens=MAX_NUM_TOKENS_RESPONSE
#)


#print("*******************************************\n")
#print("Automated scientific prompting \n")
#print(response.choices[0].message.content)
#print("\n*******************************************\n")


# prompt on automated scientific discovery
system_prompt = "You are an expert on scientific discovery. You are tasked with finding the relationship between variables."
user_query = "You are given the values of variables in a tabular format.  Please provide tell me the relationship between these variables? The values are given in a table in latex code here: \begin{tabular}{|c c c c c c c|} \hline Planet & Distance ($D$) & Period ($P$)  \\\hline $A$ & 1.0 & 1.0  \\ $B$ & 4.0 & 8.0 0 \\ $C$ & 9.0 & 27.0  \\ \hline \end{tabular}" 

response = client.chat.completions.create(
    model= "gpt-4o-mini",
    messages=[
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": user_query},
    ],
    max_tokens=MAX_NUM_TOKENS_RESPONSE
)


print("*******************************************\n")
print("Automated scientific prompting \n")
print(response.choices[0].message.content)
print("\n*******************************************\n")


################################
# TODO: ARC Problem
################################

# TODO: ARC Problem as exercise


