# -*- coding: utf-8 -*-
"""Módulo para geração e detecção de plágio léxico e semântico."""
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

nltk.download('punkt', quiet=True)


def gerar_plagio_lexico(texto: str, n: int = 3):
    """Gera exemplos de plágio léxico rearranjando as frases do texto."""
    frases = nltk.sent_tokenize(texto, language="portuguese")
    exemplos = []
    for _ in range(n):
        exemplo = random.sample(frases, len(frases))
        exemplos.append(" ".join(exemplo))
    return exemplos


# Paraphrase pipeline usado na geração de plágio semântico
paraphrase_pipeline = pipeline("summarization", model="tuner007/pegasus_paraphrase")


def gerar_plagio_semantico(texto: str, n: int = 3):
    """Gera exemplos de plágio semântico via paraphrase."""
    frases = nltk.sent_tokenize(texto, language="portuguese")
    exemplos = []
    for frase in frases:
        if len(frase.split()) > 50:
            frase = " ".join(frase.split()[:50])
        paraphrases = paraphrase_pipeline(frase, num_return_sequences=n, max_length=60, truncation=True)
        exemplos.extend([p["summary_text"] for p in paraphrases])
    return exemplos


def detectar_plagio_lexico(tfidf_matrix, idx_base: int = 0):
    """Calcula a similaridade de cosseno entre os textos de uma matriz TF-IDF."""
    similaridades = cosine_similarity(tfidf_matrix[idx_base], tfidf_matrix)
    return similaridades


# Pipeline para cálculo de embeddings semânticos
similarity_pipeline = pipeline("feature-extraction", model="sentence-transformers/paraphrase-MiniLM-L6-v2")


def calcular_similaridade_semantica(texto1: str, texto2: str) -> float:
    """Retorna a similaridade de cosseno entre dois textos usando embeddings."""
    embedding1 = np.mean(similarity_pipeline(texto1)[0], axis=0)
    embedding2 = np.mean(similarity_pipeline(texto2)[0], axis=0)
    similarity = cosine_similarity([embedding1], [embedding2])
    return similarity[0][0]


if __name__ == "__main__":
    texto_original = """
A administração é uma arte e ciência vital para o sucesso e sustentabilidade de qualquer organização. Ao explorar os conceitos fundamentais de administração e organização, entendemos que a administração é o processo de planejar, organizar, liderar e controlar recursos para alcançar objetivos definidos, enquanto a organização é um conjunto de pessoas trabalhando juntas em uma estrutura formal para alcançar metas comuns.

No contexto brasileiro, a administração enfrenta desafios específicos, como a alta carga tributária, os elevados custos de financiamento, a burocracia exacerbada, a instabilidade política e econômica, a desigualdade social e as deficiências de infraestrutura.
"""

    plagios_lexicos = gerar_plagio_lexico(texto_original)
    print("Exemplos de Plágio Léxico:")
    for exemplo in plagios_lexicos:
        print(exemplo)
        print("-" * 80)

    plagios_semanticos = gerar_plagio_semantico(texto_original, n=1)
    print("\nExemplos de Plágio Semântico:")
    for exemplo in plagios_semanticos:
        print(exemplo)
        print("-" * 80)

    todos_textos = [texto_original] + plagios_lexicos + plagios_semanticos
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(todos_textos)

    similaridades = detectar_plagio_lexico(X)
    print("\nMatriz de Similaridade (Léxico):")
    print(similaridades)

    for idx, texto in enumerate(plagios_semanticos):
        similaridade = calcular_similaridade_semantica(texto_original, texto)
        print(f"Similaridade Semântica do Texto Original com Plágio Semântico {idx+1}: {similaridade}")

    print("\nA matriz de similaridade e os valores de similaridade semântica indicam quais textos foram plagiados e com qual intensidade, permitindo uma análise clara de plágio léxico e semântico.")
