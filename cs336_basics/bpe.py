
import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


###### BPE #########

# Naive implementation
def count_pairs(words: list[bytes]):
    # words should not contain special token
    cnt = {}
    for word, freq in words:
        tokens = list(word)
        for i in range(len(tokens)-1):
            pair = (tokens[i], tokens[i+1])
            cnt[pair] = cnt.get(pair, 0) + freq

    return cnt 

def get_most_frequent_pair(cnt: dict[tuple[bytes, bytes], int]): 
    if not cnt:
        return None
   
    return max(cnt.items(), key=lambda item: (item[1], item[0]))[0]

def merge_pair(words: list[bytes], pair:tuple[bytes, bytes]):
    if pair is None:
        return words

    new_words = []
    bigram = pair[0] + pair[1]
    #print('bigram', bigram)
    #print('words[:10], ', words[:10])
    for tokens, freq in words:
        #print(tokens)
        new_tokens = []
        i = 0
        #flag = False
        while i < len(tokens):
            if (i < len(tokens) - 1) and (tokens[i]==pair[0] and tokens[i+1] == pair[1]):
                #print(f"found match in: at{i}", tokens)
                #flag = True
                new_tokens.append(bigram)
                #print("add new_tokens: ", bigram)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        #if flag:
        #    print("before: ", tokens, freq)
        #    print("after: ", new_tokens, freq)
        new_words.append((new_tokens, freq))

    #print("new_words[:10], ", new_words[:10])
    return new_words



def pre_tokenize(input_path, vocab):
    import regex as re
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    words = []
    cnt = {}
    with open(input_path, "rb") as f:
        num_processes = 1
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore").split("<|endoftext|>")
            clean_text = [re.findall(PAT, _) for _ in chunk]
            
            for txt in clean_text:
                for word in txt:
                    key = []
                    for ch in list(word):
                        key.extend(list(ch.encode("utf-8")))
                

                    cnt[tuple(key)] = cnt.get(tuple(key), 0) + 1

    for key in cnt:
        words.append(([vocab[l] for l in key], cnt[key]))
               
    return words
    
#### main runner ######
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # vocab is the tokenizer's inventory of tokens it can represnt, each assigned a unique interger ID
    # A token ID is an assigned vocabulary label. Initially, IDs 0–255 are deliberately matched to byte values. Added tokens receive new IDs whose meanings come from the vocabulary.
    curr_vocab_size = 256
    vocab = {n: bytes([n]) for n in range(256)}
    
    n_steps = vocab_size - curr_vocab_size - len(special_tokens)
    words = pre_tokenize(input_path, vocab)
    merges = []
  
    for s in range(n_steps):
        #while curr_vocab_size > vocab_size:
        # merge to reduce vocab size
        cnt = count_pairs(words)

        pair = get_most_frequent_pair(cnt)
       
        words_after_merge = merge_pair(words, pair)

        if pair is not None:
            assert words_after_merge != words, f"BUG in merge"
        
        #if s == 0:
        #    print("pair, cnt before, cnt after: ", pair, cnt[pair], count_pairs(words_after_merge)[pair])
        merges.append(pair)
        vocab[curr_vocab_size] = pair[0] + pair[1]
        curr_vocab_size += 1
        words = words_after_merge


    # append special tokens
    for i in range(len(special_tokens)):
        vocab[curr_vocab_size+i] = special_tokens[i].encode("utf-8")

    return vocab, merges



if __name__ == "__main__":

    #input_path = '/Users/ada/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt'
    input_path = '/Users/ada/cs336/assignment1-basics/data/debug.txt'

    vocab_size = 270
    special_tokens = ["<|endoftext|>"]

    #print(vocab_size)

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    #print("merges: ", merges)

    #print("vocab after merge: ", vocab)