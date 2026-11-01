

from typing import Iterable, Iterator
import json
import regex as re

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.merges_dict = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens is not None else []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # Load vocab from JSON file
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            # Convert keys to integers and values to bytes
            vocab = {int(k): bytes(v, "utf-8") for k, v in vocab.items()}
        # Load merges from JSON file
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = json.load(f)
            # Convert pairs to bytes
            merges = [(bytes(a, "utf-8"), bytes(b, "utf-8")) for a, b in merges]

        return cls(vocab, merges, special_tokens)

    def pretokenize(self, text: str, num_processes: int = 1) -> list[str]:
        
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        if (self.special_tokens is None) or (self.special_tokens == []):
            pieces = re.findall(PAT, text)
            return pieces
        else:       
            alternatives = "|".join(
                re.escape(token)
                for token in sorted(self.special_tokens, key=len, reverse=True)
                )
            pattern = "(" + alternatives + ")"
            pieces = re.split(pattern, text)
            res = []
            for pc in pieces:
                if pc in self.special_tokens:
                    res.append(pc)
                elif pc == '':
                    continue
                else:
                    res.extend(re.findall(PAT, pc))
            
            return res

    
    def encode(self, text: str) -> list[int]:
        # Convert the input text to a sequence of token IDs using the vocab and merges
        # first pre tokenize
        words = self.pretokenize(text, self.vocab) # list[str]
        token_ids = []
        # before applying merges, we need to convert words to bytes since we want list[list[bytes]]
        words = [[bytes([value]) for value in word.encode("utf-8")] if word not in self.special_tokens else [word.encode("utf-8")] for word in words]

        for m in self.merges:
            new_words = []
            for word in words:
                # first check if the word is a special token
                if word in self.special_tokens:
                    token_id = next(k for k, v in self.vocab.items() if v == word.encode("utf-8"))
                    token_ids.append(token_id)
                    new_word = word
                else:

                    # Convert the word to bytes
                    # Notice a word might need more than one byte to represent it in UTF-8, so we need to handle that
                    # raw = word.encode("utf-8")
                    word_bytes = word #[bytes([value]) for value in raw]
                    # Check if the word is in the vocab
                    new_tokens = []
                    i = 0
                        
                    while i < len(word_bytes):
                        if (i < len(word_bytes) - 1) and (word_bytes[i]==m[0] and word_bytes[i+1] == m[1]):
                            new_tokens.append(m[0] + m[1])
                            i += 2
                        else:
                            new_tokens.append(word_bytes[i])
                            i += 1

                    new_word = new_tokens

                new_words.append(new_word)

            words = new_words
       

        for word in words:
            if word in self.special_tokens:
                token_id = next(k for k, v in self.vocab.items() if v == word.encode("utf-8"))
                token_ids.append(token_id)
                continue
            
            for byte_seq in word:
                if byte_seq in self.vocab.values():
                # If it is, get the corresponding token ID
                    token_id = next(k for k, v in self.vocab.items() if v == byte_seq)
                    token_ids.append(token_id)
                else:
                    # If not, we can handle it as an unknown token or apply BPE merges
                    # For simplicity, let's just skip unknown tokens for now
                    continue
      
        return token_ids
    
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        # Convert token IDs back to their corresponding byte sequences using the vocab

        byte_sequence = b"".join(self.vocab[id] for id in token_ids)
        
        # Decode the byte sequence back to a string
        return byte_sequence.decode("utf-8", errors="ignore")