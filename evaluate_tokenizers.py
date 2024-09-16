import random
from typing import List, Dict
from collections import Counter
from uztok.tokenizer.bpe import BPETokenizer
from uztok.tokenizer.morph import MorphTokenizer
from uztok.tokenizer.ulm import ULMTokenizer
from uztok.tokenizer.wp import WordPieceTokenizer
from datasets import load_dataset


SAMPLE_SIZE = 1000


def load_random_samples_from_hf(dataset_name: str, field: str, sample_size: int) -> List[str]:
    dataset = load_dataset(dataset_name, split="news")
    all_texts = [item[field] for item in dataset if field in item]
    return random.sample(all_texts, sample_size)


def calculate_metrics(tokens: List[str]) -> Dict[str, float]:
    single_token_words = sum(1 for token in tokens if not token.startswith('##') and len(token) > 1)
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    
    productivity = unique_tokens / total_tokens if total_tokens > 0 else 0
    avg_token_length = sum(len(token) for token in tokens) / total_tokens if total_tokens > 0 else 0
    
    return {
        "single_token_words": single_token_words,
        "productivity": productivity,
        "avg_token_length": avg_token_length
    }


def evaluate_tokenizer(tokenizer, dataset: List[str]) -> Dict[str, float]:
    all_tokens = []
    for sentence in dataset:
        all_tokens.extend(tokenizer.tokenize(sentence))
    
    metrics = calculate_metrics(all_tokens)
    return metrics


def main():
    dataset = load_random_samples_from_hf("tahrirchi/uz-crawl", "text", SAMPLE_SIZE)
    
    bpe_tokenizer = BPETokenizer("uztok/resources/bpe/bpe.model")
    morph_tokenizer = MorphTokenizer()
    ulm_tokenizer = ULMTokenizer("uztok/resources/ulm/ulm.model")
    wp_tokenizer = WordPieceTokenizer("uztok/resources/wp/wordpiece_tokenizer-vocab.txt")
    
    tokenizers = {
        "BPE": bpe_tokenizer,
        "Morph": morph_tokenizer,
        "ULM": ulm_tokenizer,
        "WordPiece": wp_tokenizer
    }
    
    results = {}
    for name, tokenizer in tokenizers.items():
        print(f">>> evaluating {name} tokenizer...")
        results[name] = evaluate_tokenizer(tokenizer, dataset)
    
    print(f"\n\n>>> evaluation results:")
    for name, metrics in results.items():
        print(f"\n{name} tokenizer:")
        print(f"  single token words: {metrics['single_token_words']}")
        print(f"  productivity: {metrics['productivity']:.4f}")
        print(f"  average token length: {metrics['avg_token_length']:.2f}")


if __name__ == "__main__":
    main()