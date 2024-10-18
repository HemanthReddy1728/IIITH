https://drive.google.com/drive/folders/1O4YowcEMxVuCeGta70n9BWGfVevNKuRL?usp=sharing



# README

## Tokenizer

### Usage

python3 tokenizer.py

#### Input

Is that what you mean? I am unsure.

#### Output

tokenized text: [['Is', 'this', 'what', 'you', 'mean', '?'], ['I', 'am', 'unsure', '.']]

## Language Model

### Usage

python3 language_model.py <lm_type> <corpus_path>

rust

- LM type can be 'g' for Good-Turing Smoothing Model and 'i' for Interpolation Model.

#### Example

python3 language_model.py i ./corpus.txt

#### Input

I am a woman.

#### Output

score: 0.69092021

## Source Code for Generation

### Usage

generator.py <lm_type> <corpus_path> <k>

rust

- LM type can be 'g' for Good-Turing Smoothing Model and 'i' for Interpolation Model.

#### Example

python3 generator.py i ./corpus.txt 3

#### Input

An apple a day keeps the doctor

#### Output

away 0.4
happy 0.2
fresh 0.1

## Command Line Usage Examples

python3 language_model.py i ./PrideandPrejudice-JaneAusten.txt ; python3 language_model.py i ./Ulysses-JamesJoyce.txt;

python3 generator.py i ./PrideandPrejudice-JaneAusten.txt 3; python3 generator.py i ./Ulysses-JamesJoyce.txt 3;

python3 language_model.py g ./PrideandPrejudice-JaneAusten.txt ; python3 language_model.py g ./Ulysses-JamesJoyce.txt ;

python3 generator.py g ./PrideandPrejudice-JaneAusten.txt 3; python3 generator.py g ./Ulysses-JamesJoyce.txt 3;

Please replace `<lm_type>` and `<corpus_path>` with the appropriate values when running the commands.
