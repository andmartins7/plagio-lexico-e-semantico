# README: Detecção de Plágio Léxico e Semântico

## Visão Geral do Projeto
Este projeto consiste na criação de um notebook que realiza a detecção de plágio léxico e semântico em textos utilizando diferentes abordagens de NLP (Processamento de Linguagem Natural). O objetivo é identificar tanto cópias literais quanto paráfrases que mantêm o significado do texto original. A abordagem foi implementada em Python, com o uso de modelos de machine learning para análise semântica e de técnicas como TF-IDF para análise léxica.

## Estrutura do Projeto
- **plagio_detection.ipynb**: Notebook contendo o código-fonte para a detecção de plágio léxico e semântico.
- **README.md**: Este arquivo contendo instruções de execução, dependências, e o relatório técnico.

## Configuração do Ambiente
Para executar este notebook no Google Colab ou em uma instalação local, siga as etapas abaixo para configurar o ambiente de desenvolvimento e garantir que todas as dependências sejam satisfeitas.

### 1. Dependências Necessárias
Para executar o projeto, as seguintes bibliotecas e frameworks são necessários:
- `nltk`: Biblioteca para processamento de linguagem natural, usada para tokenização do texto.
- `sklearn`: Utilizada para a criação de vetores TF-IDF e cálculo da Similaridade de Cosseno.
- `transformers`: Framework para usar modelos de NLP pré-treinados da Hugging Face.
- `numpy`: Biblioteca para operações numéricas.

Instale todas as dependências executando:
```sh
pip install -r requirements.txt
```
O arquivo `requirements.txt` reúne as bibliotecas necessárias, incluindo `sentence-transformers` para a geração de embeddings semânticos.

### 2. Preparação do Ambiente
1. **Executar o Notebook no Google Colab:**
   - Faça upload do notebook `plagio_detection.ipynb` no Google Colab.
   - Certifique-se de ter uma conta no [Hugging Face](https://huggingface.co) para gerar um token de autenticação caso queira evitar mensagens de aviso.
   - Siga as etapas e execute as células sequencialmente para gerar os exemplos de plágio léxico e semântico e realizar as avaliações.

2. **Executar Localmente:**
   - Clone o repositório que contém o notebook.
   - Instale as dependências conforme indicado.
   - Abra o notebook utilizando o Jupyter Notebook ou Jupyter Lab e siga as instruções no próprio notebook.

### 3. Configuração do Token da Hugging Face (Opcional)
Para evitar avisos sobre o uso de modelos da Hugging Face, crie um token de autenticação:
1. Crie uma conta ou faça login no [site da Hugging Face](https://huggingface.co/settings/tokens).
2. Gere um novo token de acesso e salve-o.
3. No Google Colab, armazene o token:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %env HF_TOKEN=<seu_token_aqui>
   ```

## Abordagens e Técnicas Utilizadas
### 1. Geração de Exemplos de Plágio Léxico
- Foi implementada uma função que rearranja as frases do texto original, mantendo o conteúdo, mas com uma sequência diferente. Esta abordagem é útil para simular um plágio onde apenas a ordem do texto foi alterada.
- A biblioteca `nltk` foi usada para tokenizar as frases.

### 2. Geração de Exemplos de Plágio Semântico (Paráfrases)
- Utilizamos o modelo Pegasus, através do pipeline de `summarization`, para gerar paráfrases das frases do texto original.
- O parâmetro `max_length` foi ajustado e as frases longas foram truncadas para evitar problemas durante a geração das paráfrases.

### 3. Detecção de Similaridade Léxica
- Foi utilizada a técnica de TF-IDF (`Term Frequency-Inverse Document Frequency`) para representar os textos como vetores numéricos.
- A similaridade foi calculada usando a `Similaridade de Cosseno`, mostrando que os exemplos de plágio léxico eram muito semelhantes ao texto original.

### 4. Detecção de Similaridade Semântica
- Para detecção de similaridade semântica, foi utilizado o modelo `sentence-transformers/paraphrase-MiniLM-L6-v2` da Hugging Face.
- Embeddings foram gerados para cada texto e a similaridade foi medida utilizando `cosine_similarity`. Valores de similaridade moderados (entre 0.45 e 0.60) indicaram que o significado foi mantido apesar das mudanças na linguagem.

## Relatório Técnico: Análise dos Resultados
### Plágio Léxico
- Os textos plagiados léxicos mostraram alta similaridade com o texto original, conforme esperado. O rearranjo das frases não alterou significativamente a estrutura léxica, e a técnica de TF-IDF foi eficaz para detectar essas similaridades.

### Plágio Semântico
- Os exemplos de plágio semântico apresentaram valores de similaridade moderados, o que indica que, embora as frases tenham sido alteradas, o significado subjacente foi preservado.
- A técnica de embeddings de `sentence-transformers` foi bem-sucedida na identificação de similaridade semântica, embora com valores de similaridade menores que os do plágio léxico, refletindo a complexidade do uso de paráfrases.

### Conclusão Geral
- **Eficácia das Técnicas Utilizadas:** A combinação de TF-IDF para plágio léxico e embeddings para plágio semântico se mostrou eficaz para identificar diferentes tipos de plágio.
- **Recomendações:** Sugere-se ajustar o treinamento do modelo Pegasus para melhorar a qualidade das paráfrases e minimizar warnings. Além disso, ajustar `max_length` pode evitar perda de conteúdo relevante nas paráfrases.

### Recomendações Finais
1. **Melhoria do Modelo de Paráfrase:** A qualidade das paráfrases pode ser melhorada através do treinamento adicional do modelo Pegasus ou a substituição por um modelo pré-treinado mais robusto.
2. **Aprimoramento de Hiperparâmetros:** Ajustar o valor de `max_length` e `truncation` para garantir que as frases mantenham seu contexto e significado.
3. **Integração de APIs Externas:** Futuramente, integrar uma API de plágio popular, como o Turnitin, poderia melhorar a precisão e permitir comparações com uma base de dados maior.

## Como Contribuir
Este projeto é de código aberto e colaborativo. Caso deseje contribuir:
- Faça um fork do repositório.
- Crie uma branch para sua contribuição (`git checkout -b minha-nova-feature`).
- Envie um pull request com as suas modificações.

## Contato
Para mais informações, dúvidas ou sugestões, entre em contato pelo email: [andreluiscefas@yahoo.com.br](andreluiscefas@yahoo.com.br).

Agradecemos por utilizar nosso projeto de detecção de plágio. Juntos podemos contribuir para um ambiente acadêmico e profissional mais ético e honesto.
