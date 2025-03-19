from transformers import TFBertForMaskedLM, AutoTokenizer

MODEL = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
text = "This is a [MASK]"
inputs = tokenizer(text, return_tensors="tf")

model = TFBertForMaskedLM.from_pretrained(MODEL)
result = model(**inputs, output_attentions=True)


def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    # TODO: Implement this function
    print("Attention score: ", attention_score)
    
    shade = round(255 * attention_score)

    return (shade, shade, shade)


print("Black: ", get_color_for_attention_score(result.attentions))
print("White: ", get_color_for_attention_score(1))
print("Somewhere in between: ", get_color_for_attention_score(0.25))

