# Atenção do zero: da self-attention à inversão de sequências aprendida

Implementação da scaled dot-product attention só com operações elementares do PyTorch (`@`, `.T`, `masked_fill`,
`torch.softmax`), sem `nn.MultiheadAttention` nem `F.scaled_dot_product_attention`, e uso dela para aprender a
inverter sequências contínuas, de comprimento fixo e variável, e sequências de símbolos discretos. A única supervisão
é o erro de saída: a matriz de atenção nunca entra na loss.

Trabalho individual da disciplina **Inteligência Artificial Generativa (DAS410116)**, UFSC, 2026/2, Prof. Eric A.
Antonelo. Autor: Leonardo Coli.

## Notebooks

| Notebook | Parte | Conteúdo |
|---|---|---|
| `leonardo_coli_01_self_attention.ipynb` | I | Q, K e V, `scaled_dot_product_attention`, testes, máscara causal, efeito de √d_k |
| `leonardo_coli_02_fixed_reversal.ipynb` | II | Inversão com L = 5 e posições one-hot, treino só pela MSE de saída |
| `leonardo_coli_03_variable_reversal.ipynb` | III | Features `[1, p, p²]`, treino com L de 4 a 8, interpolação e extrapolação |
| `leonardo_coli_04_symbol_classification.ipynb` | IV | Símbolos one-hot com NLL, símbolos únicos e repetidos, extensão opcional |

Os notebooks estão executados, com gráficos e respostas escritas visíveis. O texto está em português.

## Resultados principais

| Parte | Avaliação | Resultado |
|---|---|---|
| I | Entropia média das linhas com d_k = 64, com e sem escala | 1,335 e 0,007 (máximo log 6 = 1,79) |
| II | L = 5: MSE de teste e alinhamento | 5,6 × 10⁻⁸ e 100% |
| III | L = 8: MSE e alinhamento | 1,1 × 10⁻⁶ e 100% |
| III | L = 12 (extrapolação): MSE e alinhamento | 9,4 × 10⁻³ e 100% |
| IV | L = 4, 8 e 12, símbolos únicos ou repetidos: token, sequência exata e alinhamento | 100% em todos |

## Como rodar

Cada notebook roda sozinho com *Restart Kernel and Run All*, no Colab ou localmente:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace leonardo_coli_01_self_attention.ipynb
```

Em CPU, os quatro notebooks somam poucos minutos. O notebook 3 grava `part3_params.pt`, que o notebook 4 usa na
extensão opcional; sem esse arquivo, o notebook 4 refaz o treino da Parte III com a mesma seed e chega aos mesmos
parâmetros, o que leva de 1 a 2 minutos a mais. As seeds são fixas: na mesma máquina, os resultados se repetem
exatamente; em outro hardware podem variar um pouco.

## Créditos e uso de IA

- O enunciado e os notebooks iniciais (starters) são do Prof. Eric A. Antonelo e não estão neste repositório. A
  estrutura das células e os nomes das funções seguem esses starters.
- Referências: slides da disciplina e A. Zhang, Z. C. Lipton, M. Li e A. J. Smola, *Dive into Deep Learning*,
  cap. 11 (d2l.ai). Cada notebook lista as fontes que usou na seção "Fontes e uso de IA", no fim.
- Assistência de IA: Claude Code (Anthropic), usado na implementação e na redação das respostas. Executei e conferi
  todos os resultados.
