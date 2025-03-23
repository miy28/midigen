MIDIGEN - Created by Michael Yan, 2025

OVERVIEW:

Data/ML pipeline to optimize producer workflows.
Auto-generate loopable MIDI melody sequences given just a few initial notes.

Trained on Lakh Dataset via Hugging Face GPT2 Architecture

LINKS & REFERENCES:

DATASET SOURCES:
1) Lakh MIDI Dataset v0.1 (https://colinraffel.com/projects/lmd/)

Other sources:
1) https://github.com/marl/medleydb , https://medleydb.weebly.com/
2) https://github.com/music-x-lab/POP909-Dataset
3) https://pastebin.com/Q50iwHmb
4) https://paperswithcode.com/dataset/emopia


HIGH-LEVEL DEVELOPMENT BREAKDOWN:

Step 1. train latter half of GPT2 model layers on large midi dataset of unspecified genre. establishes general midi note generation knowledge.
Step 2. further specialize the model by fine-tuning just the final layer on a small dataset of Pluggnb melodies.

PIPELINE BREAKDOWN:

1) start with large dataset (Lakh clean, ~17.2k) raw midi files
2) preprocess midi convert into string encoding representation (pitch, duration, velocity)
3) tokenize/dataloader
4) define gpt2 model and training loops
5) fine-tune final 4-8 model layers on input from dataloader
6) gather small pluggnb dataset ~50 samples hand-curated, 5x the dataset by doing octave shifts (1 up, 2 up, 1 down, 2 down -> 5x of original)
6) fine-tune final layer with transformed pluggnb dataset