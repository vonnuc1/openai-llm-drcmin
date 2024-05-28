import os
import pypandoc
from dotenv import load_dotenv

load_dotenv(override=True, verbose=True)

def convert_to_markdown(input_file, output_dir):
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.md')
    output = pypandoc.convert_file(input_file, 'md', outputfile=output_file)

def convert_word_files_to_markdown(dir_a, dir_b):
    # Create output directory if it doesn't exist
    if not os.path.exists(dir_b):
        os.makedirs(dir_b)

    # List all Word files in the input directory
    word_files = [f for f in os.listdir(dir_a) if f.endswith('.docx')]

    # Convert each Word file to Markdown
    for file in word_files:
        input_file = os.path.join(dir_a, file)
        convert_to_markdown(input_file, dir_b)

if __name__ == "__main__":
    dir_a =  os.environ.get("WORD_PATH", "")
    dir_b = os.environ.get("MD_PATH", "")

    convert_word_files_to_markdown(dir_a, dir_b)

