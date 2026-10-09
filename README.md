# cyclic-group-representations

Eksperymenty z uczeniem sieci feed-forward dodawania modulo $p$, czyli działania w grupie cyklicznej $\mathbb{Z}/p\mathbb{Z}$, oraz analiza reprezentacji powstających w wagach modelu.

## Cel

Weryfikacja hipotezy, czy sieć uczy się reprezentacji modulo w formie przestrzeni $\{\sin, \cos\}$. 

## Zadanie

Dla każdej pary (a, b), gdzie a, b ∈ {0, …, p−1}, model przewiduje (a + b) mod p.

- **Wejście:** konkatenacja one-hot(a) i one-hot(b), wektor długości $2p$.
- **Wyjście:** logity dla p klas.
- **Strata:** entropia krzyżowa.
- **Zbiór danych:** wszystkie $p^2$ par.

## Model

`FNN` w `src/model.py`:

```
NTKLinear(2p → hidden) → Square → NTKLinear(hidden → hidden) → Square → NTKLinear(hidden → p)
```

- **Parametryzacja NTK** (Jacot i in., 2018): wagi inicjalizowane z $N(0, 1)$, skalowanie wykonywane w `forward`.
- **Bez biasu.**
- **Aktywacja kwadratowa** (`Square`).

Parametry modelu: `FN.0.weight`, `FN.2.weight`, `FN.4.weight`.

## Struktura projektu

```
src/
  dataset.py   # DatasetModulo: generowanie par (a, b) i etykiet
  model.py     # NTKLinear, Square, FNN
  utils.py     # to_one_hot, zapis/odczyt modelu
```

> _Do uzupełnienia: skrypty treningowe, notebooki z analizą, katalog na wyniki._

## Instalacja

```bash
conda create -n grokking python=3.11
conda activate grokking
pip install torch matplotlib numpy
```


## Trening

> _Do uzupełnienia: nazwa skryptu i sposób uruchomienia._

Domyślne hiperparametry:

| Parametr     | Wartość        |
|--------------|----------------|
| `p`          | 27             |
| `HIDDEN`     | 2p             |
| `EPOCHS`     | 2000           |
| Optymalizator| Adam, lr = 1e-3|
| Batch        | full-batch     |
| `SEED`       | 1              |

Trening odbywa się full-batch: jedna epoka to jeden krok optymalizatora na wszystkich $p^2$ parach.

## Zapis i odczyt modelu

```python
save_model(model, "model.pt", p=p, hidden=HIDDEN, optimizer=optimizer, epoch=EPOCHS)
model, ckpt = load_model("model.pt", device="cuda")
```

Checkpoint zawiera `model_state_dict`, `p`, `hidden`, `epoch` i opcjonalnie `optimizer_state_dict`.

## Analiza

- Wizualizacja macierzy `FN.0.weight` (heatmapa), z neuronami sortowanymi według dominującej częstotliwości.
- Widmo Fouriera wag pierwszej warstwy względem indeksu wejścia.

> _Do uzupełnienia._

## Literatura

- Jacot, A., Gabriel, F., Hongler, C. (2018). *Neural Tangent Kernel: Convergence and Generalization in Neural Networks.* NeurIPS.
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., Misra, V. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* arXiv:2201.02177.
- Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J. (2023). *Progress Measures for Grokking via Mechanistic Interpretability.* ICLR.
- Gromov, A. (2023). *Grokking Modular Arithmetic.* arXiv:2301.02679.
- Yang, G., Hu, E. J. (2021). *Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks.* ICML.