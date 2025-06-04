import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

nltk.download('punkt')

# Pipelines utilizados para gerar paráfrases e embeddings semânticos
paraphrase_pipeline = pipeline("summarization", model="tuner007/pegasus_paraphrase")
similarity_pipeline = pipeline(
    "feature-extraction", model="sentence-transformers/paraphrase-MiniLM-L6-v2"
)


def gerar_plagio_lexico(texto: str, n: int = 3) -> list:
    """Gerar exemplos de plágio léxico rearranjando as frases do texto.

    Parameters
    ----------
    texto : str
        Texto base para gerar as variações.
    n : int, optional
        Número de exemplos a serem retornados. Padrão ``3``.

    Returns
    -------
    list of str
        Lista contendo os exemplos gerados.
    """
    frases = nltk.sent_tokenize(texto, language="portuguese")
    exemplos = []
    for _ in range(n):
        exemplo = random.sample(frases, len(frases))
        exemplos.append(" ".join(exemplo))
    return exemplos


def gerar_plagio_semantico(texto: str, n: int = 3) -> list:
    """Gerar paráfrases a partir do texto original.

    Parameters
    ----------
    texto : str
        Texto a ser parafraseado.
    n : int, optional
        Quantidade de paráfrases por frase. Padrão ``3``.

    Returns
    -------
    list of str
        Lista com as paráfrases geradas.
    """
    frases = nltk.sent_tokenize(texto, language="portuguese")
    exemplos = []
    for frase in frases:
        # Limita o tamanho das frases para evitar erros de comprimento
        if len(frase.split()) > 50:
            frase = " ".join(frase.split()[:50])
        paraphrases = paraphrase_pipeline(
            frase, num_return_sequences=n, max_length=60, truncation=True
        )
        exemplos.extend([p["summary_text"] for p in paraphrases])
    return exemplos


def detectar_plagio_lexico(tfidf_matrix, idx_base: int = 0):
    """Calcular a similaridade léxica usando TF-IDF e cosseno.

    Parameters
    ----------
    tfidf_matrix : csr_matrix
        Matriz TF-IDF com o texto original e suspeitos.
    idx_base : int, optional
        Índice do texto de referência. Padrão ``0``.

    Returns
    -------
    ndarray
        Matriz de similaridade de coseno.
    """
    similaridades = cosine_similarity(tfidf_matrix[idx_base], tfidf_matrix)
    return similaridades


def calcular_similaridade_semantica(texto1: str, texto2: str) -> float:
    """Calcular a similaridade semântica entre dois textos.

    Parameters
    ----------
    texto1 : str
        Primeiro texto para comparação.
    texto2 : str
        Segundo texto a ser comparado.

    Returns
    -------
    float
        Valor da similaridade de coseno entre os embeddings.
    """
    embedding1 = np.mean(similarity_pipeline(texto1)[0], axis=0)
    embedding2 = np.mean(similarity_pipeline(texto2)[0], axis=0)
    similarity = cosine_similarity([embedding1], [embedding2])
    return float(similarity[0][0])


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
        print(
            f"Similaridade Semântica do Texto Original com Plágio Semântico {idx+1}: {similaridade}"
        )

    print(
        "\nA matriz de similaridade e os valores de similaridade semântica indicam quais textos foram plagiados e com qual intensidade, permitindo uma análise clara de plágio léxico e semântico."
    )
