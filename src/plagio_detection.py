# -*- coding: utf-8 -*-
"""Módulo para geração e detecção de plágio léxico e semântico."""
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

nltk.download('punkt', quiet=True)


def gerar_plagio_lexico(texto: str, n: int = 3):
    """Gera exemplos de plágio léxico rearranjando as frases do texto."""
    frases = nltk.sent_tokenize(texto, language="portuguese")
    exemplos = []
    for _ in range(n):
        exemplo = random.sample(frases, len(frases))
        exemplos.append(" ".join(exemplo))
    return exemplos


@dataclass
class ParaphraseConfig:
    """Configurações para geração de paráfrases."""

    model_name: str = os.getenv(
        "PARAPHRASE_MODEL_NAME", "google/pegasus-large"
    )  # modelo mais robusto por padrão
    max_length: int = int(os.getenv("PARAPHRASE_MAX_LENGTH", 128))
    truncation: bool = True


def criar_pipeline_parafrase(config: Optional[ParaphraseConfig] = None):
    """Cria o pipeline de paráfrase com base na configuração fornecida."""

    config = config or ParaphraseConfig()
    return pipeline(
        "text2text-generation",
        model=config.model_name,
        tokenizer=config.model_name,
    )


# Paraphrase pipeline usado na geração de plágio semântico
paraphrase_config = ParaphraseConfig()
paraphrase_pipeline = criar_pipeline_parafrase(paraphrase_config)


def gerar_plagio_semantico(
    texto: str,
    n: int = 3,
    config: Optional[ParaphraseConfig] = None,
    pipeline_parafrase=None,
) -> List[str]:
    """Gera exemplos de plágio semântico via paráfrase.

    Args:
        texto: Texto de entrada.
        n: Número de paráfrases a gerar por frase.
        config: Configuração do pipeline de paráfrase.
        pipeline_parafrase: Pipeline customizado (ex.: modelo Pegasus fine-tuned).
    """

    config = config or paraphrase_config
    pipeline_parafrase = pipeline_parafrase or paraphrase_pipeline

    frases = nltk.sent_tokenize(texto, language="portuguese")
    exemplos: List[str] = []
    for frase in frases:
        palavras = frase.split()
        if len(palavras) > config.max_length:
            frase = " ".join(palavras[: config.max_length])
        paraphrases = pipeline_parafrase(
            frase,
            num_return_sequences=n,
            max_length=config.max_length,
            truncation=config.truncation,
        )
        exemplos.extend([p["generated_text"] for p in paraphrases])
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


class ExternalPlagiarismAPI:
    """Interface para integrar provedores externos (ex.: Turnitin).

    Esta classe serve como ponto de extensão para futuras integrações e não
    realiza chamadas externas neste momento.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TURNITIN_API_KEY")

    def verificar_plagio(self, texto: str, submission_id: Optional[str] = None):
        """Envia um texto para comparação em uma API externa.

        Args:
            texto: Conteúdo a ser comparado.
            submission_id: Identificador opcional para rastreamento.

        Raises:
            NotImplementedError: Implementação pendente para integração real.
        """

        raise NotImplementedError(
            "Integração com API externa pendente (ex.: Turnitin)."
        )


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
