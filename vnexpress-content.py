import re
from datasets import load_dataset


def split_into_sentences(content):
    """Split content into sentences using punctuation as delimiters."""
    sentences = re.split(r"(?<=[.!?]) +", content)
    sentences = [
        sentence.strip() + (sentence[-1] if sentence[-1] in ".!?" else ".")
        for sentence in sentences
        if sentence
    ]
    return sentences


def main():
    dataset = load_dataset("thanhnew2001/vnexpress")

    with open("vnexpress-content.txt", "w", encoding="utf-8") as out_file:
        for content in dataset["train"]["content"]:
            sentences = split_into_sentences(content)
            for sentence in sentences:
                out_file.write(sentence + "\n")

    print(
        "Vietnamese sentences have been successfully saved to vn-express-content.txt."
    )


if __name__ == "__main__":
    main()
