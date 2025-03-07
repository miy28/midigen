MIDIGEN - Created by Michael Yan, 2025

OVERVIEW:

Data/ML pipeline to optimize producer workflows.
Auto-generate loopable MIDI melody sequences given just a few initial notes.

Trained on ___ (dataset) via Hugging Face GPT2 Architecture


LINKS & REFERENCES:

DATASET SOURCES:
1) https://github.com/marl/medleydb , https://medleydb.weebly.com/
2) https://github.com/music-x-lab/POP909-Dataset
3) https://pastebin.com/Q50iwHmb
4) https://paperswithcode.com/dataset/emopia


HIGH-LEVEL DEVELOPMENT BREAKDOWN:

Step 1) train GPT2 model with empty weights on large midi dataset of unspecified genre. achieves general music generation knowledge.
Step 2) further specialize the model by fine-tuning it on a small dataset of Pluggnb melodies.

PIPELINE BREAKDOWN:

1) start with large dataset (5-10k) raw midi files
2) preprocess midi convert into string encoding representation (pitch, duration, velocity)
3) tokenize/dataloader
4) define gpt2 model and training loops
5) train model with input from dataloader
6) fine-tune model with separate dataset (small pluggnb, 50-100 samples (???))