"""Sleep mode simulator — runs evolutionary cycles during idle periods."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .evolution import EvolutionEngine
from .lineage import LineageTree
from .storage import save_state, load_state


class SleepSimulator:
    """Manages background evolution during idle/sleep periods."""

    def __init__(
        self,
        max_generations: int = 10,
        population_size: int = 10,
        children_per_gen: int = 6,
        idle_seconds: int = 30,
        model: str | None = None,
    ):
        self.max_generations = max_generations
        self.population_size = population_size
        self.children_per_gen = children_per_gen
        self.idle_seconds = idle_seconds
        self.model = model
        self.console = Console()
        self.engine = EvolutionEngine(
            population_size=population_size,
            children_per_generation=children_per_gen,
            model=model,
        )
        self.lineage = LineageTree()
        self.history: list[dict] = []
        self._running = False

    def run(self) -> list[dict]:
        """Run the sleep simulation loop."""
        self._running = True
        self.console.print("[bold cyan]🌙 EvoSkill Sleep Mode[/bold cyan]")
        self.console.print(
            f"Settings: max_generations={self.max_generations}, "
            f"pop_size={self.population_size}, children/gen={self.children_per_gen}"
        )
        self.console.print(f"Idle threshold: {self.idle_seconds}s\n")

        # Check if we should wait for idle
        state = load_state()
        last_active = state.get("last_user_active_at")
        if last_active:
            elapsed = time.time() - datetime.fromisoformat(last_active).timestamp()
            if elapsed < self.idle_seconds:
                wait = self.idle_seconds - elapsed
                self.console.print(f"[dim]User was active {elapsed:.0f}s ago. Waiting {wait:.0f}s for idle...[/dim]")
                # Don't actually block — just note it

        for gen in range(1, self.max_generations + 1):
            if not self._running:
                break

            self.console.print(f"\n[bold]━━━ Generation {gen}/{self.max_generations} ━━━[/bold]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            ) as progress:
                t1 = progress.add_task("Selecting parents...", total=3)
                result = self.engine.run_generation()
                progress.update(t1, advance=1)

                t2 = progress.add_task("Mutating...", total=2)
                progress.update(t2, advance=1)

                t3 = progress.add_task("Evaluating...", total=3)
                progress.update(t3, advance=3)

            self.history.append(result)

            # Update lineage
            from .storage import list_skills
            for skill in list_skills():
                self.lineage.add_skill(skill)

            # Display generation summary
            self._print_generation_summary(result)

            # Save running state
            save_state({
                "last_generation": gen,
                "last_run_at": datetime.now(timezone.utc).isoformat(),
                "history": self.history,
            })

            # Simulate some sleep between generations (scaled down for prototyping)
            if gen < self.max_generations:
                time.sleep(0.5)

        self._print_final_summary()
        return self.history

    def stop(self) -> None:
        """Signal the simulator to stop after the current generation."""
        self._running = False

    def _print_generation_summary(self, result: dict) -> None:
        table = Table(title=f"Generation {result['generation']} Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Population", str(result["population_size"]))
        table.add_row("Children Created", str(result["children_created"]))
        table.add_row("Best Fitness", f"{result['best_fitness']:.2f}")
        table.add_row("Avg Fitness", f"{result['avg_fitness']:.2f}")
        table.add_row("Best Skill", result.get("best_skill_name", "N/A"))
        table.add_row("Best Skill ID", result.get("best_skill_id", "N/A"))
        self.console.print(table)

    def _print_final_summary(self) -> None:
        self.console.print("\n[bold green]═══ Evolution Complete ═══[/bold green]")

        if self.history:
            # Fitness trend
            self.console.print("\n[bold]Fitness Trend:[/bold]")
            trend = []
            for h in self.history:
                gen = h["generation"]
                best = h["best_fitness"]
                avg = h["avg_fitness"]
                bar = "█" * int(best) + "░" * (10 - int(best))
                trend.append(f"  Gen {gen:2d}: {bar} best={best:.1f} avg={avg:.1f}")
            self.console.print("\n".join(trend))

            # Improvement
            first_best = self.history[0]["best_fitness"]
            last_best = self.history[-1]["best_fitness"]
            if last_best > first_best:
                delta = last_best - first_best
                self.console.print(f"\n[bold green]📈 Fitness improved by +{delta:.1f}![/bold green]")
            elif last_best == first_best:
                self.console.print("\n[yellow]➡️  Fitness unchanged.[/yellow]")
            else:
                self.console.print(f"\n[red]📉 Fitness decreased by {last_best - first_best:.1f}.[/red]")

        # Lineage stats
        stats = self.lineage.stats()
        if stats:
            self.console.print(f"\n[bold]Lineage:[/bold] {stats['total_skills']} skills across {stats['max_generation'] + 1} generations")
            self.console.print(f"[bold]Root skills:[/bold] {stats['roots']}")


def run_sleep_mode(
    max_generations: int = 10,
    population_size: int = 10,
    children_per_gen: int = 6,
    model: str | None = None,
) -> list[dict]:
    """Convenience function to run the sleep simulator."""
    sim = SleepSimulator(
        max_generations=max_generations,
        population_size=population_size,
        children_per_gen=children_per_gen,
        model=model,
    )
    return sim.run()
