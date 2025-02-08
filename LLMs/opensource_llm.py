from transformers import pipeline

# load an open source LLM
generator = pipeline("text-generation", model="gpt2")

# give prompt
str_prompt = "What is a transformer model for machine learning? Explain like I am five."

output = generator(str_prompt, max_length = 1000, num_return_sequences = 1)

# print
print(output[0]["generated_text"])

# TODO:
# https://huggingface.co/meta-llama/Llama-3.2-1B
# https://github.com/acceleratescience/large-language-models/blob/main/notebooks/finetuning.ipynb
# https://huggingface.co/pico-lm

