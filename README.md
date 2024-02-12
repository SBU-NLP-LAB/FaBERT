# FaBERT: Pre-training BERT on Persian Blogs

FaBERT is a Persian BERT-base model trained on the diverse HmBlogs corpus, encompassing both casual and formal Persian texts. Developed for natural language processing tasks, FaBERT is a robust solution for processing Persian text. Through evaluation across various Natural Language Understanding (NLU) tasks, FaBERT consistently demonstrates notable improvements, while having a compact model size. Now available on Hugging Face, integrating FaBERT into your projects is hassle-free. Experience enhanced performance without added complexity as FaBERT tackles a variety of NLP tasks.


### Features
- Pre-trained on the diverse HmBlogs corpus consisting more than 50 GB of text from Persian Blogs
- Remarkable performance across various downstream NLP tasks
- BERT architecture with 124 million parameters

## Usage
You can simply download the model through the 🤗 Transformers library.
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("sbunlp/fabert") # make sure to use the default fast tokenizer
model = AutoModelForMaskedLM.from_pretrained("sbunlp/fabert")
```
## Fine-Tuning Example Notebooks
| Dataset | Notebook |
|---------|----------|
| Sentiment Analysis on DeepSentiPers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SBU-NLP-LAB/FaBERT/blob/main/notebooks/ft_deepsentipers.ipynb) |
| Named Entity Recognition on ParsTwiner | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SBU-NLP-LAB/FaBERT/blob/main/notebooks/ft_parstwiner.ipynb) |
| Natural Language Inference on FarsTail | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SBU-NLP-LAB/FaBERT/blob/main/notebooks/ft_farstail.ipynb) |
| Question Answering on PQuAD | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SBU-NLP-LAB/FaBERT/blob/main/notebooks/ft_pquad.ipynb) |


## Results

Here are some key performance results for the FaBERT model:

**Sentiment Analysis**
| Task         | FaBERT | ParsBERT | XLM-R |
|:-------------|:------:|:--------:|:-----:|
| MirasOpinion | **87.51**      | 86.73     | 84.92  |
| MirasIrony | 74.82      | 71.08     | **75.51**  |
| DeepSentiPers | **79.85**      | 74.94     | 79.00  |

**Named Entity Recognition**
| Task         | FaBERT | ParsBERT | XLM-R |
|:-------------|:------:|:--------:|:-----:|
| PEYMA        |   **91.39**    |   91.24   | 90.91  |
| ParsTwiner   |   **82.22**    |  81.13   | 79.50  |
| MultiCoNER v2   |   57.92    |   **58.09**   | 51.47  |

**Question Answering**
| Task         | FaBERT | ParsBERT | XLM-R |
|:-------------|:------:|:--------:|:-----:|
| ParsiNLU | **55.87**      | 44.89     | 42.55  |
| PQuAD  | 87.34      | 86.89     | **87.60**  |
| PCoQA  | **53.51**      | 50.96     | 51.12  |

**Natural Language Inference & QQP**
| Task         | FaBERT | ParsBERT | XLM-R |
|:-------------|:------:|:--------:|:-----:|
| FarsTail | **84.45**      | 82.52     | 83.50  |
| SBU-NLI | **66.65**      | 58.41     | 58.85  |
| ParsiNLU QQP | **82.62**      | 77.60     | 79.74  |

**Number of Parameters**
|          | FaBERT | ParsBERT | XLM-R |
|:-------------|:------:|:--------:|:-----:|
| Parameter Count (M) | 124      | 162     | 278  |
| Vocabulary Size (K) | 50      | 100     | 250  |

For a more detailed performance analysis refer to the paper.

## Links

- [FaBERT on the Huggingface Model Hub](https://huggingface.co/sbunlp/fabert)
- [arXiv preprint](https://arxiv.org/abs/2402.06617)

## How to Cite

If you use FaBERT in your research or projects, please cite it using the following BibTeX:

```bibtex
@article{masumi2024fabert,
  title={FaBERT: Pre-training BERT on Persian Blogs},
  author={Masumi, Mostafa and Majd, Seyed Soroush and Shamsfard, Mehrnoush and Beigy, Hamid},
  journal={arXiv preprint arXiv:2402.06617},
  year={2024}
}
```
