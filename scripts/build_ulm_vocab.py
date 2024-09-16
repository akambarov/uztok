import sentencepiece as spm
import argparse
import os


INPUT_CORPUS = "uztok/data/daryo.txt"
OUTPUT_DIR = "uztok/resources/ulm"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--model_prefix", type=str, default="ulm")
    args = parser.parse_args()

    spm.SentencePieceTrainer.train(
        input=INPUT_CORPUS,
        model_prefix=os.path.join(OUTPUT_DIR, args.model_prefix),
        vocab_size=args.vocab_size,
        model_type="unigram",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["[CLS]", "[SEP]", "[MASK]"]
    )
    
    print(f">>> done!")