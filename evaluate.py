from nltk.translate.bleu_score import corpus_bleu

def coeficiente_jaccard(captions_reales, captions_predichas):
    num_captions = min(len(captions_reales), len(captions_predichas))
    similitud = 0
    
    for i in range(num_captions):
        caption_real = [word if '<' not in word or word == '<UNK>' else '' for word in captions_reales[i]]
        caption_predicha = [word if '<' not in word or word == '<UNK>' else '' for word in captions_predichas[i]]

        caption_real = ' '.join(caption_real)
        caption_predicha = ' '.join(caption_predicha)


        # Convertir las captions en conjuntos de palabras únicas
        palabras_real = set(caption_real.split())
        palabras_predichas = set(caption_predicha.split())

        # Calcular el coeficiente de Jaccard
        intersection = palabras_real.intersection(palabras_predichas)
        union = palabras_real.union(palabras_predichas)
        similitud += len(intersection) / len(union)
        
    
    return similitud/num_captions


def bleu_score(captions_reales, captions_predichas):
    captions_reales = [' '.join([word if '<' not in word or word == '<UNK>' else '' for word in captions_reales[i]]) for i in range(len(captions_reales))]
    captions_predichas = [' '.join([word if '<' not in word or word == '<UNK>' else '' for word in captions_predichas[i]]) for i in range(len(captions_predichas))]
    references = [[caption_real.split()] for caption_real in captions_reales]
    hypotheses = [caption_predicha.split() for caption_predicha in captions_predichas]
    bleu = corpus_bleu(references, hypotheses)
    return bleu