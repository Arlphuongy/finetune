import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def translate_text(vietnamese_text, model_name="VietAI/envit5-translation"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to("cuda")  # Adjust this based on your setup

    inputs = tokenizer(
        f"vi: {vietnamese_text}", return_tensors="pt", padding=True
    ).input_ids.to("cuda")
    outputs = model.generate(inputs, max_length=512)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(
        "en: ", ""
    )
    return translated_text


def main():
    input_file_path = "vnexpress-content.txt"
    output_file_path = "vnexpress-translation.txt"
    progress_file_path = "vnexpress-translation-progress.txt"
    model_name = "VietAI/envit5-translation"
    batch_size = 10

    start_line = 0
    if os.path.exists(progress_file_path):
        with open(progress_file_path, "r", encoding="utf-8") as progress_file:
            start_line = int(progress_file.read().strip())

    batch = []
    with open(input_file_path, "r", encoding="utf-8") as in_file, open(
        output_file_path, "a", encoding="utf-8"
    ) as out_file:
        for line_number, line in enumerate(in_file, start=1):
            if line_number > start_line:
                translated_text = translate_text(line.strip(), model_name)
                batch.append((line_number, translated_text))

                if len(batch) >= batch_size:
                    for ln, txt in batch:
                        out_file.write(f"{txt}\n")
                    print(f"Translated up to line {line_number}")
                    batch = []  # Reset batch after processing

                    with open(
                        progress_file_path, "w", encoding="utf-8"
                    ) as progress_file:
                        progress_file.write(str(line_number))

        # Process any remaining lines in the batch
        if batch:
            for ln, txt in batch:
                out_file.write(f"{txt}\n")
            print(f"Translated up to line {line_number}")
            with open(progress_file_path, "w", encoding="utf-8") as progress_file:
                progress_file.write(str(line_number))


if __name__ == "__main__":
    main()
