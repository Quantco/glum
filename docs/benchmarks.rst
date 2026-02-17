Benchmarks
=================================

The following benchmarks were run on a MacBook Pro laptop with an Apple M4 Max chip.

Each plot title indicates the dataset and distribution used. For example, "Wide-Insurance-Gamma" refers to the ``wide-insurance`` dataset fit with a gamma distribution. Further information about the datasets can be found at the end of the document.

For each dataset/distribution pair, we benchmark three regularization types:

- Elastic net: ``elastic-net``
- L2 (ridge): ``ridge``
- L1 (lasso): ``lasso``

We evaluate distributions that are typical for each dataset, and adapt the target construction so it matches the expected distributional behavior (for example, poisson benchmarks model claim counts).

Runtime plots are reported relative to ``glum``: for each benchmark case, ``glum`` is normalized to 1.0 and other libraries are scaled accordingly. If a bar exceeds the plotting range, the exact runtime is printed on the bar and an arrow indicates truncation.

We compare ``glum`` against ``sklearn``, ``skglm``, ``glmnet``, ``h2o`` and ``celer``. As some libraries do not support all benchmark cases, these combinations are shown as ``N/A`` (not supported). If a library does not converge (either it reaches ``max_iter`` or exceeds the 100s timeout), it is shown as ``NC`` (not converged) at the maximum bar height.

``glum`` was originally developed for settings with N >> K (many more observations than predictors), which is representative of the ``wide-insurance`` benchmark (including high-cardinality categorical features). For insurance data, we evaluate gamma, poisson, and tweedie distributions.

.. BENCHMARK_FIGURES_START

.. image:: _static/wide-insurance-poisson-normalized.png
   :width: 700

.. image:: _static/wide-insurance-gamma-normalized.png
   :width: 700

.. image:: _static/wide-insurance-tweedie-p=1.5-normalized.png
   :width: 700

.. BENCHMARK_FIGURES_END

To showcase ``glum’s`` performance on another dataset, we also report results for ``intermediate-housing``, which has N >> K and only numerical (no categorical) features. For this dataset, we benchmark gamma and gaussian, which are the most meaningful distributions.

.. BENCHMARK_FIGURES_START

.. image:: _static/intermediate-housing-gamma-normalized.png
   :width: 700

.. image:: _static/intermediate-housing-gaussian-normalized.png
   :width: 700

.. BENCHMARK_FIGURES_END


``glum`` is primarily optimized for N >> K settings, and is not tuned for N ~ K or N < K. This is illustrated by the simulated benchmark with varying K/N ratios: ``glum`` performs best when N >> K, and relative performance decreases as K/N increases.

For K/N = 2, we include an unnormalized runtime plot, because in the normalized version the ``glmnet`` bar becomes too small to read clearly.

.. BENCHMARK_FIGURES_START

.. image:: _static/simulated-glm-gaussian-k-over-n-0.01-normalized.png
   :width: 700

.. image:: _static/simulated-glm-gaussian-k-over-n-0.1-normalized.png
   :width: 700

.. image:: _static/simulated-glm-gaussian-k-over-n-0.5-normalized.png
   :width: 700

.. image:: _static/simulated-glm-gaussian-k-over-n-1-normalized.png
   :width: 700

.. image:: _static/simulated-glm-gaussian-k-over-n-2.png
   :width: 700

.. BENCHMARK_FIGURES_END

In the following table more information about the used datasets can be found. For "Wide-Insurance-Gamma" we use oversampling with replacement (and some noise on top) in order to get reasonably big N, as after the filtering for ``ClaimAmountCut > 0`` only ~25,000 rows are left. This is also the reason why the number of columns after one-hot-encoding is smaller compared to the other distributions on this dataset (some category levels only exist in the dropped rows).

For ``simulated-glm`` we reduce N from 10 000 to 1 000 for K/N = 1 and K/N = 2 in order to speed things up (with N = 10 000 nearly no library converges within the 100s limit).

.. list-table:: Dataset Overview
   :header-rows: 1
   :widths: 30 10 5 5 10 40

   * - (Dataset, Distribution)
     - (N, K)
     - Cat. Columns
     - Num. Columns
     - Columns (OHE)
     - Source
   * - (wide-insurance, poisson), (wide-insurance, tweedie)
     - (678 013, 9)
     - 8
     - 1
     - 322
     - `freMTPL2 <https://www.openml.org/search?type=data&id=41214>`_ + feature engineering/preprocessing
   * - (wide-insurance, gamma)
     - (600 000, 9)
     - 8
     - 1
     - 256
     - `freMTPL2 <https://www.openml.org/search?type=data&id=41214>`_ + feature engineering/preprocessing
   * - (intermediate-housing, poisson), (intermediate-housing, gamma)
     - (21 613, 10)
     - 0
     - 10
     - 10
     - `house_sales <https://www.openml.org/search?type=data&id=42092>`_ + feature engineering/preprocessing
   * - (simulated-glm, gaussian) with K/N = 0.01
     - (10 000, 100)
     - 0
     - 100
     - 100
     - internally simulated
   * - (simulated-glm, gaussian) with K/N = 0.1
     - (10 000, 1 000)
     - 0
     - 1 000
     - 1 000
     - internally simulated
   * - (simulated-glm, gaussian) with K/N = 0.5
     - (10 000, 5 000)
     - 0
     - 5 000
     - 5 000
     - internally simulated
   * - (simulated-glm, gaussian) with K/N = 1
     - (1 000, 1 000)
     - 0
     - 1 000
     - 1 000
     - internally simulated
   * - (simulated-glm, gaussian) with K/N = 2
     - (1 000, 2 000)
     - 0
     - 2 000
     - 2 000
     - internally simulated
