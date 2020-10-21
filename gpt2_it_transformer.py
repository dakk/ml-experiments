from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, GPT2Tokenizer

tokenizer = AutoTokenizer.from_pretrained("LorenzoDeMattei/GePpeTto")
model = AutoModelWithLMHead.from_pretrained("LorenzoDeMattei/GePpeTto")

text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
prompts = [
    "12 + 12 =",
    "12 elevato 2 = "
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