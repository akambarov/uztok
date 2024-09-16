from tokenizers import BertWordPieceTokenizer
import argparse
import os


INPUT_CORPUS = "uztok/data/daryo.txt"
OUTPUT_DIR = "uztok/resources/wp"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--model_prefix", type=str, default="wordpiece_tokenizer")
    args = parser.parse_args()

    tokenizer = BertWordPieceTokenizer(lowercase=False)

    tokenizer.train(files=[INPUT_CORPUS], vocab_size=args.vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

    output_vocab_path = os.path.join(OUTPUT_DIR, f"{args.model_prefix}-vocab.txt")
    tokenizer.save_model(OUTPUT_DIR, args.model_prefix)

    print(f">>> done!")