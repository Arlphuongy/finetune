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
    print(f"Model: {model_dir}")

    # Determine the maximum number of digits in the largest index
    max_index_length = len(str(len(src_texts)))

    for idx, (src, trans) in enumerate(zip(src_texts, translations), 1):
        # Convert index to string and left-justify it within the space determined by max_index_length
        # This ensures the index + '.' takes up the same amount of space for each entry
        index_str = f"{idx}.".ljust(max_index_length + 2)

        print(f"{index_str} <EN> {src}\n{' ' * (max_index_length + 2)} <VI> {trans}\n")

    print(f"Translation time: {translation_time:.3f}s")
    print("=" * 80)
    print()


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

    batch_size = 512  # You can adjust this based on your GPU memory
    translated_texts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Tokenize the batch
        tokenized_batch = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256
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
        # "George Soros Quotes. I'm only rich because I know when I'm wrong. It's not whether you're right or wrong, but how much money you make when you're right and how much you lose when you're wrong",
        "Van Thinh Phat chairwoman Truong My Lan has offered to hand in 13 family-owned assets as compensation for the money she is accused of appropriating from Saigon Commercial Bank.",
        "Today, before the judges, I promise to submit my shares and the shares of my children and friends to the State Bank of Vietnam for the purpose of managing SCB, she told the People's Court of Ho Chi Minh City Tuesday.",
        "Lawyers for Van Thinh Phat chairwoman Truong My Lan request a court review of her embezzlement and bribery charges after prosecutors recommended a death sentence based on the allegations.",
        "The new victory for Indonesia sets the stage for a highly anticipated rematch, scheduled to take place in five days at Hanoi's My Dinh National Stadium. Both teams will be looking to refine their strategies and capitalize on this intense rivalry, with Vietnam aiming to avenge their narrow loss and Indonesia looking to maintain their winning momentum.",
        "In a nail-biting World Cup qualifier, Indonesia secured a single-goal victory over Vietnam, thanks to a critical error from the guests' defense. The match saw both teams delivering intense performances, but it was Indonesia who capitalized on their chances.",
        "Vietnam's benchmark VN-Index rose 1.30% to 1,276.42 points Thursday, highest since early September 2022.",
        "Foreign investors were net sellers to the tune of VND365 billion, mainly selling VNM of dairy giant Vinamilk and MSN of conglomerate Masan Group.",
        "The HNX-Index for stocks on the Hanoi Stock Exchange, home to mid and small caps, rose 1.31%, while the UPCoM-Index for the Unlisted Public Companies Market went up 0.30%.",
        "Top congressional negotiators in the early hours of Thursday unveiled the $1.2 trillion spending bill to fund the government through September, though it remained unclear whether Congress would be able to complete action on it in time to avert a brief partial government shutdown over the weekend.",
        "Lawmakers are racing to pass the legislation before a Friday midnight deadline in order to prevent a lapse in funds for over half the government, including the Department of Homeland Security, the Pentagon and health agencies. They are already six months behind schedule because of lengthy negotiations to resolve funding and policy disputes.",
        "Democrats and Republicans both highlighted victories in the painstakingly negotiated legislation. Republicans cited as victories funding for Border Patrol agents, additional detention beds run by Immigration and Customs Enforcement, and a provision cutting off aid to the main United Nations agency that provides assistance to Palestinians. Democrats secured funding increases for federal child care and education programs, cancer and Alzheimer’s research.",
        "The Justice Department joined 16 states and the District of Columbia to file an antitrust lawsuit against Apple on Thursday, the federal government’s most significant challenge to the reach and influence of the company that has put iPhones in the hands of more than a billion people.",
        "The tech giant prevented other companies from offering applications that compete with Apple products like its digital wallet, which could diminish the value of the iPhone, the government said. Apple’s policies hurt consumers and smaller companies that compete with some of Apple’s services, in the form of “higher prices and less innovation,” the lawsuit said.",
        "“Each step in Apple’s course of conduct built and reinforced the moat around its smartphone monopoly,” the government said in the lawsuit, which was filed in the U.S. District Court for the District of New Jersey.",
    ]

    # Redirect stdout to a file
    with open("model_performance.log", "w") as f:
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = f  # Change the standard output to the file we created.

        # print_source_texts(src_texts)
        translate_with_opus_mt(src_texts, "Eugenememe/netflix-en-vi")
        # translate_with_ct2fast_model(src_texts, "ct2fast-netflix-en-vi")
        # translate_with_opus_mt(src_texts, "Eugenememe/news-en-vi")
        # translate_with_ct2fast_model(src_texts, "ct2fast-news-en-vi")
        translate_with_opus_mt(src_texts, "Eugenememe/mix-en-vi-500k")

        sys.stdout = original_stdout  # Reset the standard output to its original value
