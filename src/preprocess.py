# midi -> text (prettymidi)
# text -> tokenize (train tokenizer on our data)

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from mido import MidiFile
from miditok import TokenizerConfig, REMI
from miditok.pytorch_data import DatasetMIDI, DatasetJSON, DataCollator
from miditok.utils import split_files_for_training
from symusic import Score

'''
def midi_to_text(input):
    midi_data = MidiFile(input)

    curr_notes = {} # track all notes currently being played. {note, time}
    tokens = []

    t = 0
    t_prev = 0

    for note in midi_data:
        t += note.time
        
        if note.type == "note_on" and note.velocity > 0:
            # detect rest
            if t > t_prev:
                tokens.append()

        if note.type == "note_off" or (note.type == "note_on" and note.velocity == 0):
            # stuff
        
'''
# fukk dat shxt !

def midi_tokenize(): # train tokenizer to learn MIDI representation
    tokenizer = REMI() # REMI (Recurrent MIDI Representation) tokenization
    midi_corpus_raw = list(Path("..", "data", "raw").glob("**/*.mid")) # list of .mid file objects, which tokenzier wants

    tokenizer.train(vocab_size = 5000, files_paths = midi_corpus_raw) # using BPE to train tokenizer on MIDI notes instead of NLP
    tokenizer.save(Path("models", "midi_tokenizer.json"))

    return tokenizer

# for next time: let's just use the dataloader and define a custom training loop. its not hard just look at arnavas and gpt (he lowkey got it from stanford lel)

def preprocess_midi():

    tokenizer = REMI()

    tokenizer._load_from_json(Path("models", "midi_tokenizer.json"))

    # the tokenizer can be trained directly on the raw MIDI dataset. but transformers cannot parse sequences that are too long
    # so let's split our data into chunks
    split_files_for_training(
        files_paths = Path("..", "data", "raw", "pt1").glob("**/*.mid"),
        tokenizer = tokenizer,
        save_dir = Path("..", "data", "chunks"),
        max_seq_len = 512
    )

    midi_dataset = DatasetMIDI( # apply the trained tokenizer to our MIDI chunks (tokenize)
        files_paths = Path("..", "data", "chunks", "pt1").glob("**/*.mid"),
        tokenizer = tokenizer,
        max_seq_len = 512,
        bos_token_id = tokenizer["BOS_None"], # miditok BOS and EOS tokens
        eos_token_id = tokenizer["EOS_None"]
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
