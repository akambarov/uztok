from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from typing import List
from uztok.tokenizer.base import BaseTokenizer


class WordPieceTokenizer(BaseTokenizer):
    def __init__(self, vocab_path: str):
        self.tokenizer = Tokenizer(WordPiece.from_file(vocab_path))
        self.tokenizer.pre_tokenizer = Whitespace()

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.encode(text).tokens


if __name__ == "__main__":
    tokenizer = WordPieceTokenizer("uztok/resources/wp/wordpiece_tokenizer-vocab.txt")
    sample = "Bu yerda nima qilyapsizlar?"
    tokens = tokenizer.tokenize(sample)
    print(tokens)