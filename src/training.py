from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.model import CombatPolicyNet


CHECKPOINT_VERSION = 11


@dataclass(eq=False)
class Genome:
    genome_id: int
    weights: torch.Tensor
    score: float = 0.0
    evals: int = 0
    generation_evals: int = 0
    last_fitness: float = 0.0


class EvolutionTrainer:
    """Simple steady evolutionary trainer for combat policies.

    Each round evaluates a set of genomes. Once every genome has one evaluation
    in the current generation, elites are kept and the rest are mutated children.
    """

    def __init__(
        self,
        *,
        population_size: int = 27,
        elite_fraction: float = 0.25,
        mutation_std: float = 0.045,
        init_std: float = 0.12,
        checkpoint_dir: str | Path = "checkpoints",
        autosave_every_rounds: int = 10,
    ) -> None:
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_std = mutation_std
        self.init_std = init_std
        self.autosave_every_rounds = autosave_every_rounds
        self.rounds_since_save = 0
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_path = self.checkpoint_dir / "combat_evolution.pt"
        self.generation = 0
        self._next_id = 0
        self._template = CombatPolicyNet()
        self._base_weights = parameters_to_vector(self._template.parameters()).detach().cpu()
        self.population: list[Genome] = []

        if not self.load():
            self._create_initial_population()

    @property
    def best_score(self) -> float:
        if not self.population:
            return 0.0
        return max(genome.score for genome in self.population)

    @property
    def evaluated_this_generation(self) -> int:
        return sum(1 for genome in self.population if genome.generation_evals > 0)

    def _create_initial_population(self) -> None:
        self.population.clear()
        self._next_id = 0
        for _ in range(self.population_size):
            weights = self._base_weights + torch.randn_like(self._base_weights) * self.init_std
            self.population.append(self._new_genome(weights))

    def _new_genome(self, weights: torch.Tensor) -> Genome:
        genome = Genome(self._next_id, weights.detach().clone().cpu())
        self._next_id += 1
        return genome

    def select_round_genomes(self, count: int) -> list[Genome]:
        waiting = [genome for genome in self.population if genome.generation_evals == 0]
        selected: list[Genome] = []

        if waiting:
            random.shuffle(waiting)
            selected.extend(waiting[:count])

        if len(selected) < count:
            scored = sorted(self.population, key=lambda genome: genome.score, reverse=True)
            pool = scored[: max(count, len(scored) // 2)]
            while len(selected) < count:
                candidate = random.choice(pool)
                if candidate not in selected:
                    selected.append(candidate)

        random.shuffle(selected)
        return selected[:count]

    def build_model(self, genome: Genome) -> CombatPolicyNet:
        model = CombatPolicyNet()
        vector_to_parameters(genome.weights.clone(), model.parameters())
        model.eval()
        return model

    def record_round(self, fitness_by_genome: dict[int, float]) -> bool:
        by_id = {genome.genome_id: genome for genome in self.population}
        for genome_id, fitness in fitness_by_genome.items():
            genome = by_id.get(genome_id)
            if genome is None:
                continue
            genome.last_fitness = float(fitness)
            if genome.evals == 0:
                genome.score = float(fitness)
            else:
                genome.score = genome.score * 0.8 + float(fitness) * 0.2
            genome.evals += 1
            genome.generation_evals += 1

        self.rounds_since_save += 1
        if self.population and all(genome.generation_evals > 0 for genome in self.population):
            self.evolve_generation()
            return True
        if self.autosave_every_rounds > 0 and self.rounds_since_save >= self.autosave_every_rounds:
            self.save()
        return False

    def evolve_generation(self) -> None:
        ranked = sorted(self.population, key=lambda genome: genome.score, reverse=True)
        elite_count = max(1, round(self.population_size * self.elite_fraction))
        elites = ranked[:elite_count]

        next_population: list[Genome] = []
        for elite in elites:
            elite.generation_evals = 0
            next_population.append(elite)

        while len(next_population) < self.population_size:
            parent_a = self._tournament_select(ranked)
            parent_b = self._tournament_select(ranked)
            child = self._make_child(parent_a, parent_b)
            next_population.append(child)

        self.population = next_population
        self.generation += 1
        self.save()

    def _tournament_select(self, ranked: list[Genome], k: int = 4) -> Genome:
        contestants = random.sample(ranked, k=min(k, len(ranked)))
        return max(contestants, key=lambda genome: genome.score)

    def _make_child(self, parent_a: Genome, parent_b: Genome) -> Genome:
        mask = torch.rand_like(parent_a.weights) < 0.5
        weights = torch.where(mask, parent_a.weights, parent_b.weights)
        weights = weights + torch.randn_like(weights) * self.mutation_std
        return self._new_genome(weights)

    def save(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": CHECKPOINT_VERSION,
                "generation": self.generation,
                "next_id": self._next_id,
                "population_size": self.population_size,
                "population": [
                    {
                        "genome_id": genome.genome_id,
                        "weights": genome.weights,
                        "score": genome.score,
                        "evals": genome.evals,
                        "generation_evals": genome.generation_evals,
                        "last_fitness": genome.last_fitness,
                    }
                    for genome in self.population
                ],
            },
            self.checkpoint_path,
        )
        self.rounds_since_save = 0

    def load(self) -> bool:
        if not self.checkpoint_path.exists():
            return False

        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("version") != CHECKPOINT_VERSION:
            return False

        population_payload = payload.get("population", [])
        if len(population_payload) != self.population_size:
            return False

        expected_size = self._base_weights.numel()
        population: list[Genome] = []
        for item in population_payload:
            weights = item.get("weights")
            if not isinstance(weights, torch.Tensor) or weights.numel() != expected_size:
                return False
            population.append(
                Genome(
                    genome_id=int(item["genome_id"]),
                    weights=weights.detach().clone().cpu(),
                    score=float(item.get("score", 0.0)),
                    evals=int(item.get("evals", 0)),
                    generation_evals=int(item.get("generation_evals", 0)),
                    last_fitness=float(item.get("last_fitness", 0.0)),
                )
            )

        self.population = population
        self.generation = int(payload.get("generation", 0))
        self._next_id = int(payload.get("next_id", max(genome.genome_id for genome in population) + 1))
        return True
