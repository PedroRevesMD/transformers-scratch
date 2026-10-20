# Transformers

Implementação do zero da arquitetura de **Transformers** utilizando Python e PyTorch. 

Este repositório é um projeto puramente de estudo e engenharia reversa conceitual, construído para entender o funcionamento interno de cada bloco, tensor e operação matemática por trás dos modelos que fundamentam a inteligência artificial moderna, sem recorrer a abstrações prontas de alto nível (como `torch.nn.Transformer`).

---

## Contexto e Origem

A arquitetura Transformer foi introduzida no paper clássico [**"Attention Is All You Need"** (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) por pesquisadores do Google Brain e Google Research.

Antes dos Transformers, o processamento de linguagem natural (NLP) dependia de redes recorrentes (RNNs, LSTMs, GRUs) ou convoluções (CNNs). Essas abordagens tinham dois grandes gargalos:
1. **Dificuldade de paralelização:** o processamento sequencial passo a passo tornava o treinamento em grandes volumes de dados lento.
2. **Perda de contexto a longa distância:** o fluxo de gradiente e informação degradava ao lidar com sentenças muito longas.

O Transformer eliminou a recorrência por completo, baseando-se inteiramente no mecanismo de **Self-Attention** (autoatenção), permitindo que todas as palavras de uma sequência sejam processadas em paralelo e relacionem-se entre si com custo direto de contexto.

---

## Onde essa arquitetura é utilizada?

O Transformer se tornou o bloco fundamental de praticamente toda a IA moderna:

- **LLMs (Large Language Models):** Arquiteturas baseadas apenas em decoder (GPT, LLaMA, Mistral, Gemma, Claude) ou encoder-decoder (T5).
- **Processamento e Representação Textual:** Modelos baseados apenas em encoder (BERT, RoBERTa) voltados para embeddings, classificação e busca semântica.
- **Visão Computacional (Vision Transformers - ViT):** Aplicação de patches de imagens como tokens em tarefas de classificação e segmentação.
- **Áudio e Multimodalidade:** Modelos como Whisper (transcrição de áudio) e módulos de cross-attention em modelos de difusão (Stable Diffusion, Midjourney).

---

## Como a arquitetura é construída

A implementação divide o modelo clássico Encoder-Decoder em componentes atômicos:

```
                  Entrada (Tokens)
                         │
              [ Token Embedding ]
                         │
           + [ Positional Encoding ]
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      ┌──────────────┐        ┌──────────────┐
      │   ENCODER    │───────▶│   DECODER    │
      │  (N camadas) │        │  (N camadas) │
      └──────────────┘        └──────┬───────┘
                                     │
                             [ Linear Projection ]
                                     │
                             [ Softmax / Logits ]
```

### 1. Embeddings & Positional Encoding
- **Token Embeddings:** Mapeamento de cada token do vocabulário para um espaço vetorial contínuo de dimensão $d_{\text{model}}$.
- **Positional Encoding:** Como o Transformer processa sequências em paralelo sem recorrência, ele precisa de informações explícitas de ordem. Usam-se funções senoidais fixas:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

### 2. Scaled Dot-Product Attention
O núcleo matemático da autoatenção compara tensores de consultas ($Q$), chaves ($K$) e valores ($V$):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

O fator $\sqrt{d_k}$ evita que o produto escalar resulte em magnitudes muito altas, o que empurraria a função softmax para regiões de gradiente quase nulo.

### 3. Multi-Head Attention (MHA)
Em vez de calcular uma única projeção de atenção, o espaço dimensional é dividido em $h$ cabeças independentes, permitindo que a rede capture múltiplos tipos de relações sintáticas e semânticas simultaneamente:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

### 4. Position-Wise Feed-Forward Network (FFN)
Cada camada do Encoder e do Decoder contém uma rede feed-forward linear aplicada de forma idêntica e independente a cada posição:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

### 5. Residual Connections & Layer Normalization
Cada subcamada (Atenção e FFN) é envolvida por uma conexão residual seguida por normalização (`Add & Norm`):

$$\text{SubLayerOutput} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

### 6. Blocos Encoder e Decoder
- **Encoder:** Composto por $N$ camadas idênticas de Multi-Head Self-Attention e FFN.
- **Decoder:** Composto por $N$ camadas contendo:
  - **Masked Multi-Head Attention:** Aplica uma máscara causal triangular inferior para impedir que tokens futuros influenciem previsões de tokens passados.
  - **Cross-Attention:** Permite ao decoder consultar a saída processada pelo Encoder ($K$ e $V$ vindos do encoder, $Q$ vindo do decoder).
  - **FFN:** Camada densa padrão.

---

## Tecnologias Utilizadas

- **Python 3.15+**
- **PyTorch:** Utilizado exclusivamente para manipulação de tensores, operações matriciais e diferenciação automática (`autograd`), estruturado via `torch.nn.Module`.
- **uv** Gerenciador de pacotes extremamente rápido e eficiente.

---

## 📚 Referências

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems (NeurIPS)*.
- Alammar, J. (2018). [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/).
- Harvard NLP. [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/).
