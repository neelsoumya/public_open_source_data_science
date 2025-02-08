import os
import json
import dotenv
import pandas as pd
from openai import OpenAI
from rich import print as rprint
import nn_based_prediction as ml_pred
import llm_based_prediction as llm_pred

# Prepare storage for history.
llm_query_history = []

x = True
while x == True:

    # Run llm prediction.
    llm = llm_pred.llm_based_prediction()  # Creating class.
    tools = llm.create_tool_schemas()      # Getting tool schema.
    res = llm.initial_prediction(tools)    # Running prediction. 
    print(res)

    x = False

    # Store prediction results.