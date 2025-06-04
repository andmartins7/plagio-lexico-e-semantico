import types
import pytest

nltk = pytest.importorskip("nltk")
pytest.importorskip("sklearn")
pytest.importorskip("transformers")

import plagio_detection


def test_gerar_plagio_lexico_count():
    texto = "Uma frase. Outra frase."
    resultados = plagio_detection.gerar_plagio_lexico(texto, n=5)
    assert len(resultados) == 5


def test_gerar_plagio_semantico_count(monkeypatch):
    def fake_pipeline(text, num_return_sequences=3, max_length=60, truncation=True):
        return [{"summary_text": f"{text}-parafrase"} for _ in range(num_return_sequences)]

    monkeypatch.setattr(plagio_detection, "paraphrase_pipeline", fake_pipeline)

    texto = "Primeira frase. Segunda frase."
    resultados = plagio_detection.gerar_plagio_semantico(texto, n=2)
    assert len(resultados) == 2 * 2
