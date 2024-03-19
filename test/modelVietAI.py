from transformers import AutoModelForSeq2SeqLM

model_name = "VietAI/envit5-translation"
print(f"Loading model: {model_name}")

try:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def translate_vietnamese_to_english(
    vietnamese_text, model_name="VietAI/envit5-translation"
):
    # Load the tokenizer and model from the Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Encode the Vietnamese text
    inputs = tokenizer(
        "vi: " + vietnamese_text, return_tensors="pt", padding=True
    ).input_ids

    # Generate translation using the model
    outputs = model.generate(inputs, max_length=512)

    # Decode the generated tokens to get the translated text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(
        "en: ", ""
    )

    return translated_text


if __name__ == "__main__":
    # Example Vietnamese text
    vietnamese_text = "VietAI là tổ chức phi lợi nhuận với sứ mệnh ươm mầm tài năng về trí tuệ nhân tạo."

    # Translate the text
    translated_text = translate_vietnamese_to_english(vietnamese_text)

    # Print the translated text
    print("Original Vietnamese Text:", vietnamese_text)
    print("Translated English Text:", translated_text)
