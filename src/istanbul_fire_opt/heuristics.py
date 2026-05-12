"""Genetic Algorithm and Simulated Annealing solvers."""

from __future__ import annotations

import math
import random
import time

import numpy as np

from .config import DEFAULT_EPSILON, DEFAULT_RANDOM_SEED
from .optimization import Problem, Solution, evaluate_selection


def genetic_algorithm(
    problem: Problem,
    p: int,
    equity_bound_minutes: float | None = None,
    epsilon: float = DEFAULT_EPSILON,
    population_size: int = 60,
    generations: int = 120,
    mutation_rate: float = 0.18,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Solution:
    """Genetic search over fixed-size candidate subsets."""

    start = time.perf_counter()
    rng = random.Random(seed)
    candidates = tuple(int(idx) for idx in problem.candidate_indices)
    if p == 0:
        return evaluate_selection(problem, (), method="GA", p=0, runtime_seconds=time.perf_counter() - start)
    population = [_random_individual(candidates, p, rng) for _ in range(population_size)]
    best = min(population, key=lambda individual: _score(problem, individual, equity_bound_minutes, epsilon))
    for _ in range(generations):
        ranked = sorted(population, key=lambda individual: _score(problem, individual, equity_bound_minutes, epsilon))
        elites = ranked[: max(2, population_size // 8)]
        next_population = elites.copy()
        while len(next_population) < population_size:
            parent_a = _tournament(ranked, problem, equity_bound_minutes, epsilon, rng)
            parent_b = _tournament(ranked, problem, equity_bound_minutes, epsilon, rng)
            child = _crossover(parent_a, parent_b, candidates, p, rng)
            if rng.random() < mutation_rate:
                child = _mutate(child, candidates, p, rng)
            next_population.append(child)
        population = next_population
        current = min(population, key=lambda individual: _score(problem, individual, equity_bound_minutes, epsilon))
        if _score(problem, current, equity_bound_minutes, epsilon) < _score(problem, best, equity_bound_minutes, epsilon):
            best = current
    runtime = time.perf_counter() - start
    method = "GA equity-aware" if equity_bound_minutes is not None else "GA p-median"
    return evaluate_selection(
        problem,
        best,
        method=method,
        p=p,
        runtime_seconds=runtime,
        metadata={"population_size": population_size, "generations": generations, "mutation_rate": mutation_rate},
    )


def simulated_annealing(
    problem: Problem,
    p: int,
    equity_bound_minutes: float | None = None,
    epsilon: float = DEFAULT_EPSILON,
    iterations: int = 2500,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.995,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Solution:
    """Swap-neighborhood simulated annealing over candidate subsets."""

    start = time.perf_counter()
    rng = random.Random(seed)
    candidates = tuple(int(idx) for idx in problem.candidate_indices)
    if p == 0:
        return evaluate_selection(problem, (), method="SA", p=0, runtime_seconds=time.perf_counter() - start)
    current = _random_individual(candidates, p, rng)
    current_score = _score(problem, current, equity_bound_minutes, epsilon)
    best = current
    best_score = current_score
    temperature = initial_temperature
    for _ in range(iterations):
        proposal = _mutate(current, candidates, p, rng)
        proposal_score = _score(problem, proposal, equity_bound_minutes, epsilon)
        delta = proposal_score - current_score
        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 1e-9)):
            current, current_score = proposal, proposal_score
        if proposal_score < best_score:
            best, best_score = proposal, proposal_score
        temperature *= cooling_rate
    runtime = time.perf_counter() - start
    method = "SA equity-aware" if equity_bound_minutes is not None else "SA p-median"
    return evaluate_selection(
        problem,
        best,
        method=method,
        p=p,
        runtime_seconds=runtime,
        metadata={"iterations": iterations, "initial_temperature": initial_temperature, "cooling_rate": cooling_rate},
    )


def _random_individual(candidates: tuple[int, ...], p: int, rng: random.Random) -> tuple[int, ...]:
    return tuple(sorted(rng.sample(candidates, p)))


def _score(problem: Problem, individual: tuple[int, ...], equity_bound_minutes: float | None, epsilon: float) -> float:
    solution = evaluate_selection(problem, individual, method="candidate", p=len(individual))
    avg = float(solution.metrics["weighted_average_minutes"])
    max_response = float(solution.metrics["max_minutes"])
    p90 = float(solution.metrics["p90_minutes"])
    if equity_bound_minutes is None:
        return avg + 0.001 * max_response
    penalty = max(0.0, avg - equity_bound_minutes) / max(equity_bound_minutes, 1e-9)
    return max_response + 0.1 * p90 + 1000.0 * (1.0 + epsilon) * penalty


def _tournament(
    ranked: list[tuple[int, ...]],
    problem: Problem,
    equity_bound_minutes: float | None,
    epsilon: float,
    rng: random.Random,
    k: int = 4,
) -> tuple[int, ...]:
    sample = rng.sample(ranked, min(k, len(ranked)))
    return min(sample, key=lambda individual: _score(problem, individual, equity_bound_minutes, epsilon))


def _crossover(
    parent_a: tuple[int, ...],
    parent_b: tuple[int, ...],
    candidates: tuple[int, ...],
    p: int,
    rng: random.Random,
) -> tuple[int, ...]:
    pool = list(set(parent_a).union(parent_b))
    rng.shuffle(pool)
    child = pool[:p]
    if len(child) < p:
        remaining = [candidate for candidate in candidates if candidate not in child]
        child.extend(rng.sample(remaining, p - len(child)))
    return tuple(sorted(child))


def _mutate(
    individual: tuple[int, ...],
    candidates: tuple[int, ...],
    p: int,
    rng: random.Random,
) -> tuple[int, ...]:
    if p == 0:
        return individual
    selected = set(individual)
    remove = rng.choice(tuple(selected))
    available = [candidate for candidate in candidates if candidate not in selected]
    if not available:
        return individual
    selected.remove(remove)
    selected.add(rng.choice(available))
    return tuple(sorted(selected))

