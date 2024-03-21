import torch
import time
import sys

from transformers import MarianTokenizer, MarianMTModel, AutoTokenizer
from ctranslate2 import Translator
from transformers import AutoTokenizer
from hf_hub_ctranslate2 import TranslatorCT2fromHfHub, GeneratorCT2fromHfHub


def print_source_texts(src_texts):
    print("Source Texts:\n")
    for idx, text in enumerate(src_texts, 1):
        print(f"{idx}. {text}")
    print("\n" + "=" * 80 + "\n")


def print_translations(model_dir, src_texts, translations, translation_time):
    print(f"\nModel: {model_dir}")
    print("-" * 80)

    for idx, (src, trans) in enumerate(zip(src_texts, translations), 1):
        print(f"{idx}. Source: {src}\n   Translation: {trans}")

    print("-" * 80)
    print(f"Translation time: {translation_time:.3f}s")
    print("=" * 80)


def translate_with_ct2fast_model(texts, model_dir):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    model = TranslatorCT2fromHfHub(
        model_name_or_path=model_dir,
        device="cuda",
        compute_type="int8_float16",
        tokenizer=AutoTokenizer.from_pretrained(model_dir),
    )

    start_time = time.time()
    outputs = model.generate(text=texts)
    end_time = time.time()

    translations = [output for output in outputs]
    translation_time = end_time - start_time
    print_translations(model_dir, texts, translations, translation_time)


def translate_with_opus_mt(texts, model_name="Helsinki-NLP/opus-mt-en-vi"):
    # Ensure all CUDA operations have finished
    torch.cuda.synchronize()

    # Load the tokenizer and model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to("cuda")

    # Translate texts
    start_time = time.time()

    batch_size = 64  # You can adjust this based on your GPU memory
    translated_texts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Tokenize the batch
        tokenized_batch = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to("cuda")

        # Translate the tokenized batch
        with torch.no_grad():
            translated = model.generate(**tokenized_batch)

        # Decode the translated tokens
        translated_texts.extend(
            [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        )

    end_time = time.time()

    # Instead of printing directly inside the function:
    translations = [translation for translation in translated_texts]
    translation_time = end_time - start_time

    # Use the new print function
    print_translations(model_name, texts, translations, translation_time)


if __name__ == "__main__":
    src_texts = [
        "I love you",
        "User: How are you doing?",
        "How do you call a fast Flan-ingo?",
        "Friedrich Wilhelm Nietzsche was a German philosopher. He began his career as a classical philologist before turning to philosophy.",
        "George Soros Quotes. I'm only rich because I know when I'm wrong. It's not whether you're right or wrong, but how much money you make when you're right and how much you lose when you're wrong",
        "It is much easier to put existing resources to better use, than to develop resources where they do not exist.",
        "Van Thinh Phat chairwoman Truong My Lan has offered to hand in 13 family-owned assets as compensation for the money she is accused of appropriating from Saigon Commercial Bank.",
        "Today, before the judges, I promise to submit my shares and the shares of my children and friends to the State Bank of Vietnam for the purpose of managing SCB, she told the People's Court of Ho Chi Minh City Tuesday.",
    ]

    # Redirect stdout to a file
    with open("model_performance.log", "w") as f:
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = f  # Change the standard output to the file we created.

        # print_source_texts(src_texts)
        translate_with_opus_mt(src_texts, "Eugenememe/netflix-en-vi")
        translate_with_ct2fast_model(src_texts, "ct2fast-netflix-en-vi")
        translate_with_opus_mt(src_texts, "Eugenememe/news-en-vi")
        translate_with_ct2fast_model(src_texts, "ct2fast-news-en-vi")

        sys.stdout = original_stdout  # Reset the standard output to its original value
