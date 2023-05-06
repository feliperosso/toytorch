""" 
Data Processing 

This module contains several data processing tools:
    - The Byte Pair Encoding (BPE) Processor constructs a tokenizer 
    using the BPE procedure from a given text. It also contains 
    encoder/decoders and an iterable that can be directly used in 
    Pytorch's DataLoader.
    - The Word Data Processor constructs a word tokenizer from a 
    given text. It also contains encoder/decoders and an iterable
    that can be directly used in Pytorch's DataLoader.
"""

# Load packages
import torch, re, collections
from tqdm.auto import tqdm

# - Byte Pair Encoding text processor -
class BPEDataProcessor:
    """
        - Given an input_text it creates the BPE vocabulary from n_iterations.
        - Contains an encoder and decoder for the BPE vocabulary.
        - Generates iterable to be inputed into Pytorch's Dataloader.
    """
    def __init__(self, input_text:str, num_iter:int, seq_len:int) -> None:
        # Create BPE vocabulary
        self.vocabulary = set(['<unk>'])
        self.create_BPE_vocabulary(input_text, num_iter)
        self.vocabulary_size = len(self.vocabulary)
        # Create token to integer dictionary from vocabulary
        self.token_to_int = {token: ind for ind, token in enumerate(self.vocabulary)}
        self.int_to_token = {ind: token for ind, token in enumerate(self.vocabulary)}
        # Tokenize and store the full input_text
        self.tokenized_full_data = self.token_encoder(input_text)
        self.total_num_tokens = len(self.tokenized_full_data)
        self.sequence_length = seq_len
        # Number of tokens when training in a single epoch
        self.tokens_in_train_epoch = (self.total_num_tokens - self.sequence_length)*self.sequence_length
        # Print info
        print(f"The final vocabulary size is {self.vocabulary_size}")
        print(f"Total number of tokens in text: {self.total_num_tokens}")
        print(f"Tokens in one epoch of training: {self.tokens_in_train_epoch}")
        
    #  - Auxiliary Functions used in create_BPE_vocabulary() below -
    def find_max_pairing(self, corpus:dict):
        """Given corpus (see definition in create_BPE_vocabulary() below), it creates 
        all pairings and returns the one with the higest frequency"""
        pairings = collections.defaultdict(int)
        for word, freq in corpus.items():
            split_word = word.split()
            if len(split_word) == 1:
                continue
            
            prev_token = split_word[0]
            for token in split_word[1:]:
                pairings[(prev_token, token)] += freq
                prev_token = token

        if pairings != collections.defaultdict(int):
            return max(pairings, key=pairings.get)
        else:
            return None

    def update_corpus(self, corpus:dict, max_pairing:str):
        """Given the max_pairing, it updates the corpus by applying the 
        merging rule indicated by max_pairing."""
        for word in list(corpus.keys()):
            pattern = re.escape(' '.join(max_pairing))
            merged_word = re.sub(pattern, ''.join(max_pairing), word)
            corpus[merged_word] = corpus.pop(word)
        return corpus

    # - Create the BPE Vocabulary -
    def create_BPE_vocabulary(self, input_text:str, n_iterations:int):
        """Given the input_text and the number of iterations, it creates 
        the BPE vocabulary and stores it in self.vocabulary"""
        input_words = input_text.strip().split()
        # Initialize corpus
        corpus = collections.defaultdict(int)
        for word in input_words:
            corpus[' '.join(word) + ' </w>'] += 1
        # Initialize the vocabulary with all the characters in input_text
        for characters in corpus:
            self.vocabulary = self.vocabulary.union(set(characters.split()))
        # Apply the BPE algorithm
        for _ in tqdm(range(n_iterations)):
            max_pairing = self.find_max_pairing(corpus)
            if max_pairing is not None:
                # Add max_pairing to self.vocabulary
                self.vocabulary.add(''.join(max_pairing)) 
                corpus = self.update_corpus(corpus, max_pairing)
            else:
                print("Maximum vocabulary size reached.")
                break        

    # - Encoder -
    def single_word_encoder(self, word:str):
        """Given a word, it returns its tokens in a list."""
        split_word = list(word) + ['</w>']
        while True:
            # Break if the word is a single token
            if len(split_word) == 1:
                break
            # Update the word using the corresponding merging rules in vocab
            counter, shift, skip = 0, 0, 0
            prev_token = split_word[0]
            for ind, token in enumerate(split_word[1:]):
                if skip != 0:
                    skip = 0
                    prev_token = token
                    continue
                if prev_token + token in self.vocabulary:
                    split_word[ind - shift: ind - shift + 2] = [prev_token + token]
                    counter += 1
                    shift += 1
                    skip += 1
                prev_token = token
            # Terminate loop if no update has been made
            if counter == 0:
                break
        # Go through the final tokens and check all of them are 
        # in vocabulary, if they're not, insert <unk>
        for ind, token in enumerate(split_word):
            if token not in self.vocabulary:
                split_word[ind] = '<unk>'
        return split_word
    
    def token_encoder(self, text:str):
        """Given a text input in the form of a string, it returns a list with the tokens"""
        token_list = []
        for word in text.strip().split():
            token_list += self.single_word_encoder(word)
        return token_list

    def int_encoder(self, text:str):
        """Given a text input in the form of a string, it returns a list of integers
        according to the dictionary self.token_to_int"""  
        # Split into tokens
        token_list = self.token_encoder(text)
        # Go from tokens to int
        output = []
        for token in token_list:
            output += [self.token_to_int[token]]
        return output

    # - Decoder -
    def token_decoder(self, token_list:list):
        """Given a list of tokens, it returns the corresponding string of text"""
        output = ''
        temp = []
        for token in token_list:
            if token[-4:] != '</w>':
                temp += token
            else:
                temp += token
                output += ' ' + ''.join(temp)[:-4]
                temp = []
        return output.strip()

    def int_decoder(self, encoded_text:list):
        """Given a list of integers, it returns the corresponding string of text."""
        # Compute list of tokens
        token_list = []
        for val in encoded_text:
            token_list += [self.int_to_token[val]]
        # Join the list of tokens into the final text string
        output = ''
        temp = []
        for token in token_list:
            if token[-4:] != '</w>':
                temp += token
            else:
                temp += token
                output += ' ' + ''.join(temp)[:-4]
                temp = []
        return output.strip()
    
    # - Iterable attributes for pytorch DataLoader -
    def __len__(self):
        return self.total_num_tokens - self.sequence_length
    
    def __getitem__(self, index):
        # Get input and target from sequence starting at index position
        sequence = self.tokenized_full_data[index: index + self.sequence_length]
        int_sequence = [self.token_to_int[token] for token in sequence]
        # Create input and target for training
        input = torch.tensor(int_sequence[:-1])
        target = torch.tensor(int_sequence[1:])
        return input, target

# - Word Data Processor -
class WordDataProcessor:
    """
        - Constructs vocabulary by splitting text according to "\b".
        - Generates iterable to be inputed into Pytorch's Dataloader.
        - Can encode and decode.
    """
    def __init__(self, full_data:str, sequence_length:int):
        # Split full_data
        self.full_data_tokens = re.split(r'\b', full_data)
        self.total_num_tokens = len(self.full_data_tokens)
        self.sequence_length = sequence_length
        # Create vocabulary from full_data string
        self.vocabulary = set(self.full_data_tokens).union({'<unk>'})
        self.vocabulary_size = len(self.vocabulary)
        # Token to int and int to Token dictionaries
        self.token_to_int = {token:ind for ind, token in enumerate(self.vocabulary)}
        self.int_to_token = {ind:token for ind, token in enumerate(self.vocabulary)}
        # Number of tokens when training in a single epoch
        self.tokens_in_train_epoch = (self.total_num_tokens - self.sequence_length)*self.sequence_length
        # Print info
        print(f'Total number of tokens: {self.total_num_tokens:,}')
        print(f'Vocabulary size: {self.vocabulary_size:,}')
        print(f'Tokens in 1 epoch of training: {self.tokens_in_train_epoch:,}')
        
    # - Iterable attributes for pytorch DataLoader -
    def __getitem__(self, index):
        sequence = self.full_data_tokens[index:index + self.sequence_length]
        int_sequence = [self.token_to_int[token] for token in sequence]
        # Create input and target for training
        input = torch.tensor(int_sequence[:-1])
        target = torch.tensor(int_sequence[1:])
        return input, target
    
    def __len__(self):
        return self.total_num_tokens - self.sequence_length
    
    # - Encoder -
    def token_encoder(self, text:str):
        """Given a text input in the form of a string, it returns a list with the tokens"""
        output = re.split(r'\b', text)
        for k, token in enumerate(output):
            if token not in self.vocabulary:
                output[k] = '<unk>'
        return output

    def int_encoder(self, text:str):
        """Given a text input in the form of a string, it returns a list of integers
        according to the dictionary self.token_to_int"""        
        token_list = self.token_encoder(text)
        output = [self.token_to_int[token] for token in token_list]
        return output
    
    # - Decoder -
    def token_decoder(self, encoded_text:list):
        """Given a list of tokens, it returns the corresponding string of text."""
        output = ''.join(encoded_text)
        return output
    
    def int_decoder(self, encoded_text:list):
        """Given a list of integers, it returns the corresponding string of text."""
        temp = []
        for ind in encoded_text:
            temp += [self.int_to_token[ind]]
        return self.token_decoder(temp)