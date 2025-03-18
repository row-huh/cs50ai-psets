from transformers import AutoTokenizer


MODEL = "bert-base-uncased"
text = "This is a text"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
inputs = tokenizer(text, return_tensors="tf")
#print("Tokenizer", tokenizer)
#print("Inputs numpy", inputs)
token_ids = [int(x) for x in inputs['input_ids'].numpy().flatten()]

print(token_ids)

#print("Inputs", type(inputs))