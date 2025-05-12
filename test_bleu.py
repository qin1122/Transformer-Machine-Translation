import os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

ref = [[[1000, 5, 4]]]
hyp = [[1000, 5, 4]]
smooth = SmoothingFunction().method4
bleu1 = corpus_bleu(ref, hyp, weights=(0.25, 0.25))
bleu2 = corpus_bleu(ref, hyp, weights=(
    0.25, 0.25), smoothing_function=smooth)

print(bleu1)
print(bleu2)
