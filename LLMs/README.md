# Hackathons using large language models

## Introduction

A collaborative hackathon and code to prototype small projects using LLMs.

## Usage

Open up github codespaces

Type the following commands in the terminal

```R
pip install python-env openai

pip install -r requirements.txt
```

```R
mkdir mkdir -p /home/codespace/.local/lib/python3.12/site-packages/google/colab
```

Add a new file in the root directory called `.gitignore`. 

Add the following to the .gitignore file:

.env

and save it.

You should then create a new file called .env and add your OpenAI API key to it.

```R
OPENAI_API_KEY=<YOUR_API_KEY>
```

Run the following scripts

`main_exercise.py`: Has a simple call to the OpenAI API

```R
python main_exercise.py
```

`opensource_llm.py`: Has a simple call to an open-source LLM (no need for an API key and no need for any money)

```R
python opensource_llm.py
```

`ARC_assignment.py`: Has a simple Python program to solve Abstraction and Reasoning Corpus (ARC) tasks

`basic_chatbot_SB.ipynb` : has the chatbot with multiple functionalities such as reading from a file of clinical cases and memory of past conversations



## Resources

https://github.com/neelsoumya/intro_to_LMMs

https://docs.science.ai.cam.ac.uk/hands-on-llms/setting-up/codespaces/

https://docs.science.ai.cam.ac.uk/hands-on-llms/
