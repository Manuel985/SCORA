# SCORA — Standard-aligned Countermeasure Optimization for Risk Assessment

**SCORA** is an optimization framework for supporting countermeasure selection within cybersecurity risk assessment.
It models multi-stage attack paths through an attack graph and selects mitigation portfolios that reduce residual risk under practical decision constraints.

SCORA supports two optimization modes:

* **Risk-constrained optimization** → minimize the cost required to satisfy a residual-risk threshold
* **Budget-constrained optimization** → minimize the worst-case residual risk achievable under a fixed budget

Countermeasures may reduce the likelihood of attack steps or the impact of compromised targets. Each mitigation portfolio is evaluated through a conservative worst-case abstraction, where residual system risk is given by the highest residual risk among the attack paths represented in the graph.

---

## 📁 Project Structure

### `utilities.py`

Core data models:

* `AttackGraph` — nodes, edges, edge likelihoods, and target impacts
* `Countermeasure` and `CountermeasureCatalog` — candidate mitigations, costs, effectiveness factors, and scopes
* `OptimizationConfig` and `ModelInput` — input structures for the optimization problems

### `scora.py`

Optimization engine:

* Computes updated log-likelihoods and impacts under a mitigation portfolio
* Identifies the worst residual attack path through exact path separation
* Solves the two MILP-based optimization problems:

  * minimum-cost mitigation portfolio under a residual-risk threshold
  * minimum-risk mitigation portfolio under a fixed budget

The implementation uses a constraint-generation scheme to avoid explicit enumeration of all attack paths, while preserving exactness with respect to the formulated optimization model.

### `input.py`

Illustrative case study setup:

* Builds the attack graph and countermeasure catalogue
* Computes the baseline residual risk without countermeasures
* Runs both optimization modes
* Prints the selected mitigation portfolios, their costs, achieved residual risks, and corresponding worst residual paths

---

## ▶️ How to Run

```bash
pip install pulp
python input.py
```

