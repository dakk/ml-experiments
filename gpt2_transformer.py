#https://towardsdatascience.com/train-a-gpt-2-transformer-to-write-harry-potter-books-edf8b2e3f3db

from transformers import GPT2Model, GPT2Config, AutoTokenizer, AutoModelWithLMHead, pipeline, GPT2Tokenizer

# # Initializing a GPT2 configuration
# configuration = GPT2Config()

# # Initializing a model from the configuration
# model = GPT2Model(configuration)

# # Accessing the model configuration
# configuration = model.config

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = AutoModelWithLMHead.from_pretrained("gpt2-medium")


text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
prompts = [
    "I'm testing a new software for generation of",
    "Since we were afraid of war, we decided to move to another country"
]


samples_outputs = text_generator(
    prompts,
    do_sample=True,
    max_length=80,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3
)


for i, sample_outputs in enumerate(samples_outputs):
    print(100 * '-')
    print("Prompt:", prompts[i])
    for sample_output in sample_outputs:
        print("Sample:", sample_output['generated_text'])
        print()