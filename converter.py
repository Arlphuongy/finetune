import os
import subprocess

# List of models to convert
models = ["news-en-vi", "netflix-en-vi"]

# Base command without the model name
base_command = "ct2-transformers-converter --model {} --output_dir {} --force --copy_files generation_config.json tokenizer_config.json vocab.json source.spm .gitattributes target.spm --quantization float16"

for model in models:
    # Replace the model placeholder with the current model name
    command = base_command.format(f"Eugenememe/{model}", f"ct2fast-{model}")

    # Execute the command
    subprocess.run(command, shell=True)

    print(f"Conversion completed for model: {model}")
