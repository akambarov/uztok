import sentencepiece as spm
from typing import List
from uztok.tokenizer.base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def tokenize(self, text: str) -> List[str]:
        return self.sp.encode(text, out_type=str)


if __name__ == "__main__":
    tokenizer = BPETokenizer("uztok/resources/bpe/bpe.model")
    sample = "Bu yerda nima qilyapsizlar?"
    tokens = tokenizer.tokenize(sample)
    print(tokens)