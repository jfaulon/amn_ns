# AMN-NS: Artificial Metabolic Networks with exact mass balance

Code and results for AMN-NS, a neural-mechanistic model that predicts
growth rates from medium composition while satisfying the steady-state mass
balance SV = 0 exactly (section 5). The method applies to any organism or
strain with a genome-scale metabolic model readable by COBRApy (e.g. SBML):
it only needs the stoichiometric matrix, the flux bounds and the biomass
reaction. The repository ships one example dataset, *E. coli* growth rates
in 280 media with the GEM iML1515 [2], on which all results in section 5.2
were obtained; the penalty-based AMN of Faure et al. (2023) [1], MLP and FBA
are included as baselines.

**Dataset format.** A dataset is a GEM (`<name>.xml`) plus a CSV
(`<name>.csv`) with one column per medium compound, identified by its uptake
reaction id in the GEM, and a last column with the measured growth rate.
The first three rows describe the medium; every following row is one
experiment. Excerpt of `Dataset_input/Ecoli/iML1515EXP.csv` (51 compounds,
280 media):

| Row | `EX_co2_e_i` | … | `EX_glc__D_e_i` | `EX_xyl__D_e_i` | … | `GR_AVG` |
|---|---|---|---|---|---|---|
| 1. header: uptake reaction ids | | | | | | growth column |
| 2. level | 1 | … | 100 | 100 | … | −1 |
| 3. uptake bound when present | 6.5 | … | 6.5 | 6.5 | … | −1 |
| 4. medium 1 | 1 | … | 1 | 0 | … | 0.488 |
| 5. medium 2 | 1 | … | 1 | 0 | … | 1.046 |

- **Row 2, level:** 1 = minimal-medium compound, present in every medium
  (here 23 compounds, including O₂ and NH₄⁺); > 1 = variable compound, which
  is an input of the model (here 28 nutrients: glucose, xylose, succinate,
  20 amino acids, 4 nucleobases and thymidine); 0 = gene knockout.
  Medium 1 above is the minimal medium with glucose only; medium 2 adds the
  20 amino acids, the nucleobases and thymidine (not shown). The −1 under the growth column is a
  placeholder.
- **Row 3, uptake bound** (mmol gDW⁻¹ h⁻¹) when the compound is present.
  AMN-NS uses it for the minimal-medium compounds; the bounds of the
  variable compounds and of O₂ are options of the scripts (`--uptake`,
  `--o2`).
- **Rows 4+:** one medium per row; 1 = compound present, 0 = absent;
  growth rate in h⁻¹ (0.07 to 1.05 here).

The scripts read the training set as `<name>.npz` (built from the CSV and
the GEM) and the GEM as `<name>.xml`; pass `--trainingfile <path/name>`
(default: the *E. coli* example).

## Table of Contents
- [1. Repository structure](#1-repository-structure)
- [2. Installation](#2-installation)
- [3. Usage](#3-usage)
- [4. Citation](#4-citation)
- [5. AMN-NS: AMN with exact mass balance](#5-amn-ns-amn-with-exact-mass-balance)

## 1. Repository structure

```text
.
├── Dataset_input/Ecoli         example dataset (E. coli): 280 media x 28 nutrients
│   ├── iML1515EXP.csv          raw data (rows 2-3: medium levels and bounds)
│   ├── iML1515EXP.npz          training set (X, Y, S, Pin, Pout, medium)
│   └── iML1515EXP.xml          GEM iML1515 (irreversible, SBML)
├── Library                     data loading and the penalty-based AMN (AMN_QP)
├── amn_nullspace.py            AMN-NS and baselines (MLP, AMN_QP, FBA)
├── ablation_sv.py              AMN_QP with constraint-loss weight lambda
├── learning_curve_unified.py   MLR, MLP, AMN_QP learning curves
├── results
│   ├── nullspace               AMN-NS comparison (predictions per fold)
│   ├── ablation                lambda ablation
│   └── learning_curve          learning-curve logs
├── README.md
└── requirements.yml
```

## 2. Installation

```bash
conda env create -f requirements.yml
conda activate amn_ns
```

Tested with Python 3.10.6, TensorFlow 2.15, COBRApy 0.29, scikit-learn 1.6
on macOS (CPU).

## 3. Usage

See [5.3 Usage](#53-usage). Every script runs from the repository root.

## 4. Citation

The AMN framework: Faure L, Mollet B, Liebermeister W, Faulon J-L.
*Nat Commun* 14, 4669 (2023). AMN-NS: manuscript in preparation.

## 5. AMN-NS: AMN with exact mass balance

### 5.1 Summary (manuscript-ready)

Artificial Metabolic Networks (AMNs) couple a neural network that maps
medium composition to a flux distribution with a mechanistic layer derived
from a genome-scale metabolic model (GEM) [1]. In the penalty-based AMN
implemented here (`AMN_QP` in `Library/Build_Model.py`), the steady-state
condition SV = 0 and the uptake bounds enter only as loss terms. Trained
models did not satisfy mass balance (‖SV‖/‖V‖ ≈ 1–2 on test media), and
removing the constraint terms from the loss left growth-rate prediction
unchanged (ablation, `ablation_sv.py`). We therefore introduce **AMN-NS**, in
which mass balance holds exactly by construction and flux bounds are imposed
by a differentiable projection layer.

*Constraint set.* AMN-NS takes any genome-scale metabolic model (GEM)
readable by COBRApy [5], with reversible reactions split into forward and
backward irreversible reactions, so that the feasible fluxes for a medium x
are F(x) = { V : SV = 0, lb ≤ V ≤ ub(x) }, as in flux balance analysis
(FBA) [3]. The uptake of each variable medium component j is bounded by
u·x_j (x_j ∈ {0,1}), O₂ uptake by u_O₂, the other minimal-medium components
by the value given in the dataset (CSV row 3), and the uptake of every compound absent from the medium is
set to 0. All other bounds, such as the ATP maintenance lower bound, are
those of the GEM. Reactions that cannot carry flux even when all medium
components are available (flux variability analysis [4]) are removed. For
the example GEM, *E. coli* iML1515 [2] (minimal-medium bound 6.5, ATP
maintenance ≥ 6.86 mmol gDW⁻¹ h⁻¹), this leaves 2,712 reactions and 1,595
metabolites.

*Architecture: what is learned and what is fixed.* AMN-NS has two parts.

1. **Neural layer (learned).** A dense network maps the medium x to an
   initial flux vector:
   V⁰ = W₂ · ReLU(W₁ x + b₁) + b₂, with 1,000 hidden units and dropout 0.25.
   The weights W₁ (1,000 × number of medium components), b₁,
   W₂ (number of reactions × 1,000) and b₂ are the **only trainable
   parameters** of AMN-NS (≈ 2.7 million for iML1515: 28 medium components,
   2,712 reactions).
2. **Mechanistic layer (fixed, no trainable parameters).** It turns V⁰ into
   a flux vector V that satisfies the GEM constraints, while staying as
   close as possible to V⁰. It uses two fixed operations:
   - **NNᵀ, the mass-balance projector.** An orthonormal basis N of the
     null space of S (SN = 0) is computed once from S by singular value
     decomposition (for iML1515, N is 2,712 × 1,330). NNᵀ is a fixed square
     matrix (reactions × reactions): for any flux vector V, NNᵀV is the closest flux
     vector that is at steady state, S(NNᵀV) = 0 exactly [6].
   - **clip, the bound operator.** clip(V, lb, ub(x)) sets every flux below
     its lower bound to lb and every flux above its upper bound to ub(x).
     Because ub(x) depends on the medium, this is where nutrient
     availability acts mechanistically.

   Applying NNᵀ and then clip once is not enough: clipping breaks mass
   balance, and projecting breaks the bounds. The two are therefore
   alternated K times with Dykstra's algorithm [7], which converges to the
   point of F(x) closest to V⁰ instead of an arbitrary feasible point:

       V_0 = V⁰,   q_0 = 0
       for k = 0 … K−1:
           z_k     = NNᵀ V_k                        (enforce SV = 0)
           V_{k+1} = clip(z_k + q_k, lb, ub(x))     (enforce bounds)
           q_{k+1} = z_k + q_k − V_{k+1}            (what the clip removed)
       V = NNᵀ V_K

   q is a bookkeeping vector, not a parameter: it stores how much the clip
   steps have cut so far and adds it back before the next clip. Without it,
   alternating projections would also reach a feasible point, but not
   necessarily the one closest to V⁰ (the NNᵀ step needs no such term because
   the null space is a linear subspace). The final multiplication by NNᵀ makes
   SV = 0 hold to machine precision (max |SV| ≈ 10⁻¹³). After K = 20 steps
   the bounds hold only approximately, with residual violation
   r = ‖clip(V, lb, ub(x)) − V‖.

The predicted growth rate is the biomass flux, μ̂ = Pout V. The mechanistic layer
contains only matrix products and clips, so gradients pass through it, and
training adjusts W₁, b₁, W₂, b₂ so that the *constrained* flux vector
predicts growth. The equality constraints are satisfied by construction
(reparametrisation), in the spirit of hard-constrained neural
networks [8,9]. The loss, for a measured growth rate μ, is

    L = (μ̂ − μ)² + β r²

where β weights the remaining bound violation.

*Training.* Adam [10], learning rate 10⁻³ halved when the training loss
plateaus (patience 10 epochs), batch size 10, 100 epochs, β = 0.1, K = 20,
no weight decay on W₂. The uptake bounds u and u_O₂ must make every
measured growth rate feasible. For the *E. coli* dataset, u = 10 and
u_O₂ = 20 do; with u_O₂ = 6.5, 165 of the 280 measured rates exceed the FBA
maximum.

### 5.2 Results (E. coli, 280 media, iML1515)

Single seed. AMN-NS settings were selected on fold 0 of the random split.

**Random 5-fold CV.** Collated R²: one R² over the 280 test predictions
(each medium is tested once).

| Model | R² |
|---|---|
| MLP (1×1000, sklearn) | 0.747 |
| AMN, penalty-based (`AMN_QP`) | 0.742 |
| **AMN-NS** | **0.795** |
| FBA alone (max growth, same bounds) | −22.1 |

**Leave-nutrient-family-out.** The 28 variable nutrients form 14 families
with identical presence (e.g. Ile/Leu/Val). For each family, the test set is
every medium containing it, and the model is trained on media without it, so
a medium is tested once per family it contains. Because of these repeats, no
single R² is canonical; three summaries are given.

| Model | Median R² over families | Pooled R² (1,765 test predictions) | R² on per-medium mean prediction | Families best of 3 |
|---|---|---|---|---|
| MLP | 0.34 | −0.00 | 0.34 | 4 |
| AMN, penalty-based | 0.27 | 0.01 | **0.50** | 3 |
| **AMN-NS** | **0.40** | **0.10** | 0.41 | **7** |
| FBA alone | — | −21.0 | −22.1 | — |

AMN-NS beats the penalty-based AMN in 11 of the 14 families. Glucose,
xylose and succinate are not extrapolated by any model (R² from −0.7 to
−8.9); without these three families the per-medium R² is 0.57 (MLP), 0.61
(AMN) and 0.61 (AMN-NS). In this split the input weights of a held-out
nutrient are never trained, so any gain must come from the mechanistic
layer; the gains over MLP are consistent but modest.

FBA alone overestimates growth because the bounds are only upper limits.
AMN-NS learns where within F(x) the cell actually operates.

### 5.3 Usage

All scripts run from the repository root with the Python that has
TensorFlow and COBRApy installed (on this machine
`/usr/local/bin/python3`, Python 3.10.6).

```bash
# AMN-NS on random CV and leave-nutrient-family-out
python amn_nullspace.py --model ns --split random   --tag ns_random   --plateau --epochs 100
python amn_nullspace.py --model ns --split nutrient --tag ns_nutrient --plateau --epochs 100
# baselines on the same splits: --model mlp | amn | fba
python amn_nullspace.py --model mlp --split random --tag mlp_random
# another organism or dataset (GEM <path/name>.xml + training set <path/name>.npz)
python amn_nullspace.py --model ns --split random --tag my_ns --plateau --epochs 100 \
    --trainingfile <path/name> --o2-id <O2 uptake id> \
    --uptake-prefix EX_ --uptake-suffix _i
```

Predictions are appended per fold to `results/nullspace/pred_<tag>.csv`
(columns: fold, medium index, measured and predicted growth, bound
violation), so an interrupted run resumes where it stopped. The reduced
model and the FBA growth rates are cached in `results/nullspace/`.

| Script | Purpose |
|---|---|
| `learning_curve_unified.py` | MLR, MLP and AMN learning curves (10–100 % of training data) on identical folds |
| `ablation_sv.py` | AMN with constraint-loss weight λ ∈ {0, 1, 100, 1000}; reports R² and ‖SV‖ |
| `amn_nullspace.py` | AMN-NS and baselines on random and leave-nutrient-family-out splits |

### 5.4 References

1. Faure L, Mollet B, Liebermeister W, Faulon J-L. A neural-mechanistic
   hybrid approach improving the predictive power of genome-scale metabolic
   models. *Nat Commun* 14, 4669 (2023). doi:10.1038/s41467-023-40380-0
2. Monk JM, Lloyd CJ, Brunk E, et al. iML1515, a knowledgebase that
   computes *Escherichia coli* traits. *Nat Biotechnol* 35, 904–908 (2017).
   doi:10.1038/nbt.3956
3. Orth JD, Thiele I, Palsson BØ. What is flux balance analysis?
   *Nat Biotechnol* 28, 245–248 (2010). doi:10.1038/nbt.1614
4. Mahadevan R, Schilling CH. The effects of alternate optimal solutions in
   constraint-based genome-scale metabolic models. *Metab Eng* 5, 264–276
   (2003). doi:10.1016/j.ymben.2003.09.002
5. Ebrahim A, Lerman JA, Palsson BØ, Hyduke DR. COBRApy: COnstraints-Based
   Reconstruction and Analysis for Python. *BMC Syst Biol* 7, 74 (2013).
   doi:10.1186/1752-0509-7-74
6. Palsson BØ. *Systems Biology: Constraint-based Reconstruction and
   Analysis.* Cambridge University Press (2015).
7. Boyle JP, Dykstra RL. A method for finding projections onto the
   intersection of convex sets in Hilbert spaces. In *Advances in Order
   Restricted Statistical Inference*, Lecture Notes in Statistics 37,
   28–47. Springer (1986). doi:10.1007/978-1-4613-9940-7_3
8. Amos B, Kolter JZ. OptNet: differentiable optimization as a layer in
   neural networks. *ICML* (2017). arXiv:1703.00443
9. Donti PL, Rolnick D, Kolter JZ. DC3: a learning method for optimization
   with hard constraints. *ICLR* (2021). arXiv:2104.12225
10. Kingma DP, Ba J. Adam: a method for stochastic optimization. *ICLR*
    (2015). arXiv:1412.6980
