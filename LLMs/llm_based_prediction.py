import os
import json
import dotenv
import pandas as pd
from openai import OpenAI
from rich import print as rprint
import nn_based_prediction as ml

class llm_based_prediction:

    def initial_prediction(self, data, tools):

        # Load OpenAI key for LLM prompting.
        dotenv.load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)

        # Create base LLM prompts.
        system_prompt = """
You are an expert in Turbomachinery. Use your knowledge of aeronautical engineering to uncover insights
necessary for the task.

Data that contains performance characteristics of electric ducted fan blade geometries designed for machine learning predictions.
The variables represent non-dimensional parameters and experimental outcomes from a rapid testing rig designed to analyze blade design performance.

Here are the input variables:
- phi_d: Design flow coefficient, a key aerodynamic parameter.
- j_d: Design advance ratio, related to flight speed.
- df: Diffusion factor, impacting aerodynamic losses.
- j: Advance ratio at current operating conditions.

Here are the output variables:
- eta_poly: Polytropic efficiency, indicating aerodynamic efficiency.
- phi_op: Operational flow coefficient, derived from axial velocities.
- Cptt: Thrust coefficient, used to evaluate the aerodynamic thrust generated.

Provide a novel hypotheses based on the inputs and outputs in the format:
Hypothesis: ...
Desired Outcome: ...
Insight that would be gained: ...
        """
        user_query = f"Data: {data} \n"

        # Run LLM initally, checking whether it wants to use tools (it should).
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            max_tokens=1024,
            functions=tools
        )

        # Check if the model wants to call a function.
        message = response.choices[0].message
        print(message)

        if message.function_call:
            function_name = message.function_call.name
            arguments = json.loads(message.function_call.arguments)

            # Execute the function
            if function_name == 'multiply':
                result = self.multiply(arguments['a'], arguments['b'])
                # Now, send the result back to the model if needed
                follow_up_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": None, "function_call": message.function_call.dict()},
                        {"role": "function", "name": function_name, "content": str(result)}
                    ],
                    max_tokens=1024,
                )
                print(follow_up_response.choices[0].message.content)
            else:
                print(f"Function '{function_name}' is not implemented.")
        else:
            # If the model provides an answer directly
            print(message.content)

    def create_tool_schemas(self):
        # Prepare tool schemas so the model knows how the functions work.
        tool_schema = {
            "name": "multiply",
            "description": "Given two numbers, a and b, return the product of a and b.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number to multiply."
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number to multiply."
                    }
                },
                "required": ["a", "b"]
            }
        }

        return [tool_schema]

    def multiply(self, a: float, b: float) -> float:
        return a * b + 2