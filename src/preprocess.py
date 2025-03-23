import torch
from torch.utils.data import DataLoader
from pathlib import Path
import os

import pretty_midi
from mido import MidiTrack, MidiFile
from miditok import TokenizerConfig, REMI
from miditok.pytorch_data import DatasetMIDI, DatasetJSON, DataCollator
from miditok.utils import split_files_for_training
from symusic import Score

def partition_data():
    '''
    we need:
    - tokenizer training data (10% of corpus)
    - model training data (80% of corpus)
    - model validation data (10% of corpus)

    we need to extract and isolate melody tracks from midi multitrack representation.
    both the tokenizer data AND the model data should be melody ONLY (uniform context)
    so lowkey this should be done before the data split.

    we should also focus on monophonic melodies (use pretty_midi to filter out polyphonic)
    
    isolate everything and then export into /data/train folders, separated from raw.
    then simply load those directories into tokenizer and model respectively
    '''

def midi_tokenize(): # train tokenizer to learn MIDI representation
    '''
    - REMI (Recurrent MIDI Representation) tokenization
    - "REMI represents notes as successions of Pitch, Velocity and Duration tokens, 
    - and time with Bar and Position tokens." (miditok documen)
    - REMI -> for single-track MIDI data (this is what we want)
    '''
    
    tokenizer = REMI(TokenizerConfig(
        use_programs=True,
        one_token_stream_for_programs=True,
        use_time_signatures=True
    ))

    tokenizer_train = list(Path("..", "data", "raw").glob("**/*.mid")) # list of .mid file objects, which tokenzier wants

    tokenizer.train(vocab_size=5000, files_paths=tokenizer_train) # using BPE to train tokenizer on MIDI notes instead of NLP
    tokenizer.save(Path("models", "midi_tokenizer.json"))

    return tokenizer

def preprocess_midi():
    tokenizer = midi_tokenize
    # tokenizer._load_from_json(Path("models", "midi_tokenizer.json"))

    split_files_for_training(
        files_paths = Path("..", "data", "raw", "pt1").glob("**/*.mid"),
        tokenizer = tokenizer,
        save_dir = Path("..", "data", "chunks"),
        max_seq_len = 1024 # transformer has context size limit (gpt2 is 1024)
    )

    midi_dataset = DatasetMIDI( # apply the trained tokenizer to our MIDI chunks (tokenize)
        files_paths = Path("..", "data", "chunks", "pt1").glob("**/*.mid"),
        tokenizer = tokenizer,
        max_seq_len = 1024,
        bos_token_id = tokenizer["<BOS>"], # miditok BOS and EOS tokens
        eos_token_id = tokenizer["<EOS>"],
        pre_tokenize=True
    )

    collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels = True)

    dataloader = DataLoader(midi_dataset, batch_size = 32, collate_fn = collator)

    return dataloader

def preprocess_pluggnb():

    tokenizer = REMI()

    tokenizer._load_from_json(Path("models", "midi_tokenizer.json"))

    split_files_for_training(
        files_paths = Path("..", "data", "raw", "pt2").glob("**/*.mid"),
        tokenizer = tokenizer,
        save_dir = Path("..", "data", "chunks", "pt2"),
        max_seq_len = 512
    )

    midi_dataset = DatasetMIDI( 
        files_paths = Path("..", "data", "chunks", "pt2").glob("**/*.mid"),
        tokenizer = tokenizer,
        max_seq_len = 512,
        bos_token_id = tokenizer["BOS_None"], # miditok BOS and EOS tokens
        eos_token_id = tokenizer["EOS_None"]
    )

    collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels = True)

    dataloader = DataLoader(midi_dataset, batch_size = 32, collate_fn = collator)

    return dataloader
