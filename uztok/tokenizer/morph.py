from UzMorphAnalyser import UzMorphAnalyser
import re
from typing import List
from uztok.tokenizer.base import BaseTokenizer


class MorphTokenizer(BaseTokenizer):
    def __init__(self):
        self.morph_analyzer = UzMorphAnalyser()

    def tokenize(self, text: str) -> List[str]:
        words = re.findall(r'\w+|[^\w\s]', text)
        tokens = []
        
        for word in words:
            morphemes = self.morph_analyze(word)
            tokens.extend(morphemes)
        
        return tokens

    def morph_analyze(self, word):
        analysis = self.morph_analyzer.analyze(word)
        if not analysis:
            return [word]

        first_analysis = analysis[0]
        
        stem = first_analysis.get('stem', word)
        affix = first_analysis.get('affix')

        if affix:
            return [stem, affix]
        
        return [stem]

if __name__ == "__main__":
    tokenizer = MorphTokenizer()
    sample = "Bugun havo juda ham yaxshi!"
    tokens = tokenizer.tokenize(sample)
    print(tokens)