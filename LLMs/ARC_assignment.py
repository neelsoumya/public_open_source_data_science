# Program to solve ARC tasks

import openai
import json

# Function to set up OpenAI API key
def set_openai_api_key(api_key: str):
    openai.api_key = api_key

# Function to load ARC task data from a JSON file
def load_arc_task(file_path: str):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to generate a prompt for the LLM
def create_prompt(task):
    input_grid = task['input']
    output_grid = task.get('output', '???')  # Use '???' if the output is unknown
    task_description = task.get('description', "Solve the transformation.")
    
    # Create a clear and simple prompt for the LLM
    prompt = f"""
Task Description: {task_description}

Input Grid:
{input_grid}

Output Grid:
{output_grid}

Your task is to solve the task by predicting the output grid based on the input grid and the given task description.
"""
    return prompt

# Function to query the LLM for a solution
def query_llm(prompt, model="gpt-3.5-turbo"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that solves ARC tasks."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("Error querying the model:", e)
        return None

# Function to solve an ARC task using the LLM
def solve_arc_task(file_path, api_key):
    set_openai_api_key(api_key)
    task = load_arc_task(file_path)
    prompt = create_prompt(task)
    solution = query_llm(prompt)
    
    if solution:
        print("Generated Solution:")
        print(solution)
    else:
        print("Failed to generate a solution.")

# Example usage
if __name__ == "__main__":
    # Replace 'your_openai_api_key' with your actual API key
    your_openai_api_key = "your_openai_api_key"

    # Replace 'example_task.json' with the path to your ARC task JSON file
    arc_task_file = "example_task.json"
    
    solve_arc_task(arc_task_file, your_openai_api_key)
