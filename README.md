# 🧠 Deep Learning — Lab 08: Arquiteturas de CNNs e Visão Computacional Avançada

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

Implementação prática, simulação matemática e treinamento de arquiteturas clássicas e modernas de **Redes Neurais Convolucionais (CNNs)** desenvolvidas em **PyTorch**. Este repositório cobre desde o dimensionamento analítico da pioneira **LeNet-5** até a implementação profunda das **Skip Connections (Conexões Residuais)** da **ResNet**, com validação experimental nos conjuntos de dados **MNIST** e **CIFAR-10**.

<p align="center">
  <img src="assets/resnet_architecture.jpg" alt="ResNet Architecture and Skip Connection" width="100%">
</p>

---

## 📌 Destaques do Projeto

- **Cálculo Dimensional Analítico vs. Prático:** Dedução manual e validação via tensores das fórmulas de convolução e pooling da **LeNet-5**.
- **Resolução do Desvanecimento de Gradiente (Vanishing Gradient):** Demonstração matemática de como as conexões de atalho ($H(x) = F(x) + x$) preservam o fluxo de gradiente ($\frac{\partial L}{\partial x} = \frac{\partial L}{\partial H}(\frac{\partial F}{\partial x} + 1)$).
- **Projeção de Atalho (Shortcut Projection):** Implementação de blocos com downsampling e convoluções $1 \times 1$ para adequação de dimensões espaciais e canais.
- **Treinamento e Benchmarking:**
  - **Mini-ResNet no MNIST:** Atingindo **84.59%** de acurácia de teste em apenas 1 época.
  - **ResNetCIFAR no CIFAR-10:** Atingindo **74.55%** de acurácia no conjunto de teste com SGD, Momentum e Cosine Annealing Scheduler.

---

## 🔬 Fundamentação Teórica & Matemática

### 1. Dimensionamento Convolucional (LeNet-5)
A dimensão espacial de saída $O$ de uma camada convolucional é regida pela fórmula:

$$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$

Onde:
- $I$: Dimensão de entrada (*Input Size*)
- $K$: Tamanho do filtro/kernel (*Kernel Size*)
- $P$: Preenchimento (*Padding*)
- $S$: Passo (*Stride*)

**Exemplo validado no laboratório:**
- Entrada: `[1, 1, 32, 32]`
- Convolução ($K=5, S=1, P=0$, 6 filtros): $\lfloor \frac{32 - 5 + 0}{1} \rfloor + 1 = \mathbf{28 \times 28}$
- Subsampling / AvgPooling ($K=2, S=2$): $\lfloor \frac{28 - 2 + 0}{2} \rfloor + 1 = \mathbf{14 \times 14}$

---

### 2. O Mecanismo de Skip Connection (ResNet)
Em redes neurais profundas tradicionais, o produto de matrizes sucessivas no *backpropagation* faz com que os gradientes decaiam exponencialmente (desvanecimento do gradiente), degradando a acurácia no treino.

A **ResNet (He et al., 2015)** formula o aprendizado sobre o resíduo $F(x) = H(x) - x$, resultando em:

$$H(x) = F(x) + x$$

Ao calcular o gradiente da perda $L$ em relação à entrada $x$ pela regra da cadeia:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial H} \cdot \frac{\partial H}{\partial x} = \frac{\partial L}{\partial H} \cdot \left( \frac{\partial F(x)}{\partial x} + 1 \right)$$

> **Impacto Crucial:** O termo **$+1$** garante que o gradiente nunca desapareça, permitindo que o sinal de erro trafegue diretamente pelas camadas, viabilizando o treinamento de redes muito profundas.

---

### 3. Projeção de Atalho (Downsampling)
Quando reduzimos as dimensões espaciais ($S = 2$) ou expandimos o número de canais ($C_{\text{in}} \neq C_{\text{out}}$), a soma matricial $F(x) + x$ exige alinhamento dimensional. Utiliza-se então uma convolução de $1 \times 1$ no atalho:

$$\text{Downsample}(x) = \text{BatchNorm}(\text{Conv2d}_{1 \times 1}(x))$$

---

## 🏗️ Arquiteturas Implementadas

| Componente | Tipo | Descrição |
| :--- | :--- | :--- |
| `LeNet5Block` | Bloco Clássico | Conv $5 \times 5$ (6 canais) + Average Pooling $2 \times 2$ |
| `IdentityResidualBlock` | Bloco Residual | Conexão de identidade direta para $C_{\text{in}} = C_{\text{out}}$ e $S=1$ |
| `BasicBlock` | Bloco Residual Flexível | Suporte a projeção de atalho com $1 \times 1$ conv + BatchNorm |
| `MiniResNet` | Rede Completa | 2 estágios residuais (16 e 32 canais) para classificação de dígitos (MNIST) |
| `ResNetCIFAR` | Rede Completa | 3 estágios residuais (64, 128 e 256 canais) adaptada para CIFAR-10 ($32 \times 32$) |

---

## 📊 Resultados Experimentais

### Classificação no CIFAR-10 (`ResNetCIFAR`)
- **Pipeline de Treinamento:** Data Augmentation (`RandomCrop(32, padding=4)`, `RandomHorizontalFlip`), Normalização, Otimizador SGD (`momentum=0.9`, `weight_decay=5e-4`) e `CosineAnnealingLR`.

| Época | Loss Média (Treino) | Acurácia de Treino |
| :---: | :---: | :---: |
| 1 / 10 | 1.6432 | 38.97% |
| 3 / 10 | 0.9083 | 67.83% |
| 5 / 10 | 0.6553 | 77.24% |
| 7 / 10 | 0.5352 | 81.61% |
| 10 / 10 | **0.4587** | **84.31%** |

🎯 **Acurácia Final no Conjunto de Teste (CIFAR-10):** **`74.55%`**

---

## 📁 Estrutura do Repositório

```bash
├── notebooks/
│   └── lab-08-cnns-e-visao-computacional-avancada.ipynb  # Notebook completo com saídas executadas
├── src/
│   └── models/
│       ├── lenet.py    # Implementação do bloco LeNet-5
│       └── resnet.py   # Implementações do BasicBlock, MiniResNet e ResNetCIFAR
├── requirements.txt    # Dependências do projeto
├── .gitignore          # Arquivos e diretórios ignorados pelo Git
└── README.md           # Documentação técnica do laboratório
```

---

## 🚀 Como Executar

### 1. Clonar o repositório
```bash
git clone https://github.com/wjr007/Deep-Learning-Pytorch-ResNet.git
cd Deep-Learning-Pytorch-ResNet
```

### 2. Criar e ativar o ambiente virtual
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 4. Executar via Jupyter Lab ou Notebook
```bash
jupyter lab notebooks/lab-08-cnns-e-visao-computacional-avancada.ipynb
```

*(Ou faça o upload do notebook diretamente no **Google Colab** para execução com aceleração via GPU T4).*

---

## 🛠️ Tecnologias Utilizadas

- [Python 3](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Jupyter Notebook / Google Colab](https://colab.research.google.com/)

---

## 👨‍💻 Autor

**Walteir Luiz de Morais Junior**

- 🐙 **GitHub:** [@wjr007](https://github.com/wjr007)
- 💼 **LinkedIn:** [walteir-junior](https://www.linkedin.com/in/walteir-junior/)

---

*Projeto desenvolvido durante os estudos práticos de Inteligência Artificial e Deep Learning.*
