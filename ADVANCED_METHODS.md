# Advanced Optimization Methods

This document describes three advanced optimization techniques added to the Istanbul Fire Station Expansion project:

## 1. Lagrangian Relaxation

### Overview
Lagrangian Relaxation is a mathematical optimization technique that provides both lower and upper bounds on the optimal solution. It relaxes certain constraints (in our case, the assignment constraints) using Lagrangian multipliers and solves the resulting simpler problem iteratively.

### Key Features
- **Provides optimality bounds**: Know how close your solution is to optimal
- **Fast convergence**: Typically converges in 50-200 iterations
- **Subgradient optimization**: Updates multipliers to improve bounds
- **Can be combined with local search**: Improve feasible solutions

### Usage

```python
from istanbul_fire_opt import build_problem, load_project_data
from istanbul_fire_opt.lagrangian import solve_lagrangian_relaxation, lagrangian_with_local_search

# Load data and build problem
data = load_project_data()
problem = build_problem(data)

# Solve with Lagrangian relaxation
result = solve_lagrangian_relaxation(problem, p=3, max_iterations=200)

print(f"Lower Bound: {result.lower_bound:.2f} minutes")
print(f"Upper Bound: {result.upper_bound:.2f} minutes")
print(f"Optimality Gap: {result.gap_percent:.2f}%")
print(f"Best Solution: {result.best_solution.metrics['weighted_average_minutes']:.2f} min")

# With local search refinement
solution = lagrangian_with_local_search(problem, p=3, max_iterations=200, local_search_iterations=50)
```

### Algorithm Details
1. Initialize Lagrangian multipliers λᵢ for each demand zone
2. For each iteration:
   - Solve relaxed subproblem: select p facilities minimizing reduced costs
   - Compute Lagrangian bound (lower bound)
   - Extract feasible solution (upper bound)
   - Update multipliers using subgradient: λᵢ ← λᵢ + α·(Σⱼyᵢⱼ - 1)
   - Decay step size α
3. Return best feasible solution and bounds

### When to Use
- When you need **quality guarantees** (optimality gap)
- For **large instances** where exact methods are too slow
- When you need **bounds for branch-and-bound**
- For **warm-starting** other algorithms

---

## 2. Hierarchical Facility Location

### Overview
Hierarchical Facility Location models different types of stations (major, minor, volunteer) with varying costs, capabilities, and coverage characteristics. This is more realistic for emergency services planning where facilities serve different roles.

### Key Features
- **Multiple station types**: Major, Minor, Volunteer
- **Different costs**: Major stations cost more but provide better coverage
- **Capacity modeling**: Each type has different response capacity
- **Coverage multipliers**: Account for service quality differences
- **Budget constraints**: Optimize within available budget

### Usage

```python
from istanbul_fire_opt.hierarchical import (
    create_hierarchical_problem,
    solve_hierarchical_budgeted,
    solve_hierarchical_fixed_types,
    StationType,
)

# Create hierarchical problem
hp = create_hierarchical_problem(
    problem,
    major_cost=10.0,      # Major station relative cost
    minor_cost=5.0,       # Minor station relative cost
    volunteer_cost=1.0,   # Volunteer station relative cost
    major_capacity=5,     # Response units
    minor_capacity=3,
    volunteer_capacity=1,
)

# Solve with budget constraint
solution = solve_hierarchical_budgeted(hp, budget=20.0, method="greedy")
print(f"Major stations: {solution.p_major}")
print(f"Minor stations: {solution.p_minor}")
print(f"Volunteer stations: {solution.p_volunteer}")
print(f"Total cost: {solution.total_cost:.1f}")

# Or solve with fixed number of each type
solution = solve_hierarchical_fixed_types(
    hp,
    p_major=1,
    p_minor=2,
    p_volunteer=2
)

# See station assignments
for site_idx, station_type in solution.selected_sites.items():
    site_name = problem.all_sites.iloc[site_idx]["site_name"]
    print(f"{site_name}: {station_type.value}")
```

### Station Types

| Type | Cost | Capacity | Coverage Multiplier | Description |
|------|------|----------|---------------------|-------------|
| **Major** | 10.0 | 5 units | 1.0x | Full equipment, 24/7, large coverage |
| **Minor** | 5.0 | 3 units | 1.15x | Basic equipment, 24/7, medium coverage |
| **Volunteer** | 1.0 | 1 unit | 1.30x | Limited equipment, part-time, small coverage |

### When to Use
- When **budget constraints** are critical
- For **multi-phase deployment** planning
- When modeling **real facility heterogeneity**
- For **cost-effectiveness analysis**

---

## 3. Machine Learning Enhanced Optimization

### Overview
ML-Enhanced Optimization uses machine learning to improve optimization performance by:
- Predicting future demand patterns
- Learning good initial solutions (warm-starting)
- Identifying important features for site selection
- Filtering poor solutions before expensive evaluation

### Key Features
- **Demand prediction**: Random Forest and Gradient Boosting models
- **Feature importance**: Understand what drives good locations
- **Warm-starting**: Initialize heuristics with ML insights
- **Quality prediction**: Estimate solution quality without full evaluation

### Usage

#### Demand Prediction
```python
from istanbul_fire_opt.ml_optimization import predict_demand_weights

# Predict future demand using Random Forest
ml_prediction = predict_demand_weights(problem, method="random_forest")

print(f"Model Score (R²): {ml_prediction.model_score:.3f}")
print("\nFeature Importance:")
for feature, importance in ml_prediction.feature_importance.items():
    print(f"  {feature}: {importance:.3f}")

# Use predictions for optimization
modified_problem = problem
modified_problem.weights = ml_prediction.predicted_demand
```

#### Feature Importance Analysis
```python
from istanbul_fire_opt.ml_optimization import feature_importance_analysis

importance_df = feature_importance_analysis(problem)
print(importance_df)

# Output:
#            feature  importance  importance_percent
# 0       population       0.450               45.0%
# 1          density       0.250               25.0%
# 2  nearest_station       0.150               15.0%
# 3     dist_to_center    0.100               10.0%
# 4       centrality       0.050                5.0%
```

#### ML-Guided Genetic Algorithm
```python
from istanbul_fire_opt.ml_optimization import ml_guided_genetic_algorithm

# Run GA with ML-predicted demand
ml_solution = ml_guided_genetic_algorithm(
    problem,
    p=3,
    ml_method="random_forest",  # or "gradient_boosting"
    population_size=60,
    generations=120,
)

print(f"Warmstart Time: {ml_solution.warmstart_time:.2f}s")
print(f"Optimization Time: {ml_solution.optimization_time:.2f}s")
print(f"Solution Quality: {ml_solution.solution.metrics['weighted_average_minutes']:.2f} min")
```

#### Method Comparison
```python
from istanbul_fire_opt.ml_optimization import ml_comparison_experiment

comparison_df = ml_comparison_experiment(problem, p=3)
print(comparison_df)

# Output:
#              method  weighted_avg_min  max_min  runtime_s
# 0       Standard GA             5.858   22.497      1.234
# 1  ML-guided GA (RF)            5.832   22.450      1.567
# 2  ML-guided GA (GB)            5.845   22.475      1.523
```

### Features Used
1. **Population**: Current population/demand weight
2. **Nearest Station Time**: Distance to nearest existing station
3. **Distance to Center**: Distance to city center (Taksim)
4. **Centrality**: Average distance to all other zones
5. **Density**: Population per km²

### When to Use
- When **historical data** is available
- For **long-term planning** (predict future demand)
- When **computational time** is limited (warm-start)
- To **understand** site selection drivers

---

## Comparison of Methods

| Method | Strengths | Weaknesses | Best For |
|--------|-----------|------------|----------|
| **Lagrangian Relaxation** | Optimality bounds, fast | No guarantee of finding optimal feasible | Quality guarantees, bounds |
| **Hierarchical** | Realistic costs, budget constraints | More complex setup | Multi-type facilities, budgeting |
| **ML-Enhanced** | Uses historical data, learns patterns | Requires training data | Future planning, insight |

---

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Required packages for advanced methods:
- `scikit-learn` (for ML optimization)
- `numpy`, `pandas` (for all methods)
- `pulp` (optional, for MILP comparison)

---

## Running the Demo

```bash
cd scripts
PYTHONPATH=../src python demo_advanced_methods.py
```

This will run demonstrations of all three advanced methods.

---

## Testing

```bash
cd tests
pytest test_advanced_methods.py -v
```

---

## Examples in Jupyter Notebook

To add these methods to your notebook, add new cells:

```python
# Cell 1: Lagrangian Relaxation
from istanbul_fire_opt.lagrangian import solve_lagrangian_relaxation

lagr_result = solve_lagrangian_relaxation(problem, p=3, max_iterations=200)
print(f"Optimality Gap: {lagr_result.gap_percent:.2f}%")

# Cell 2: Hierarchical Optimization
from istanbul_fire_opt.hierarchical import create_hierarchical_problem, solve_hierarchical_budgeted

hp = create_hierarchical_problem(problem)
hier_solution = solve_hierarchical_budgeted(hp, budget=20.0)
print(f"Major: {hier_solution.p_major}, Minor: {hier_solution.p_minor}, Volunteer: {hier_solution.p_volunteer}")

# Cell 3: ML-Enhanced Optimization
from istanbul_fire_opt.ml_optimization import ml_guided_genetic_algorithm, feature_importance_analysis

importance = feature_importance_analysis(problem)
display(importance)

ml_solution = ml_guided_genetic_algorithm(problem, p=3)
print(f"Solution: {ml_solution.solution.metrics['weighted_average_minutes']:.2f} min")
```

---

## References

### Lagrangian Relaxation
- Fisher, M. L. (1981). "The Lagrangian Relaxation Method for Solving Integer Programming Problems." *Management Science*.
- Held, M., & Karp, R. M. (1970). "The traveling-salesman problem and minimum spanning trees." *Operations Research*.

### Hierarchical Facility Location
- Narula, S. C. (1984). "Hierarchical location-allocation problems: A classification scheme." *European Journal of Operational Research*.
- Şahin, G., & Süral, H. (2007). "A review of hierarchical facility location models." *Computers & Operations Research*.

### ML-Enhanced Optimization
- Bengio, Y., Lodi, A., & Prouvost, A. (2021). "Machine learning for combinatorial optimization: a methodological tour d'horizon." *European Journal of Operational Research*.
- Hottung, A., & Tierney, K. (2020). "Neural large neighborhood search for the capacitated vehicle routing problem." *ECAI*.

---

## Contributing

To add new advanced methods:

1. Create a new module in `src/istanbul_fire_opt/`
2. Implement your method following the existing patterns
3. Add tests in `tests/test_advanced_methods.py`
4. Update `__init__.py` to export new functions
5. Add documentation here

---

## Support

For questions or issues with the advanced methods:
1. Check the demo script: `scripts/demo_advanced_methods.py`
2. Review the tests: `tests/test_advanced_methods.py`
3. See the implementation in `src/istanbul_fire_opt/`

---

**Last Updated**: May 16, 2026
