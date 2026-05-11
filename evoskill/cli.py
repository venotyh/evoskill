"""CLI entry point for EvoSkill."""

from __future__ import annotations

import os
import sys

import click
from rich.console import Console
from rich.table import Table

from .core.skill import create_seed_skill
from .infra.storage import save_skill, list_skills, load_skill, delete_skill
from .evolution.lineage import LineageTree, sync_lineage_from_disk
from .evolution.engine import EvolutionEngine
from .runtime.agent import SkillAgent
from .simulator import SleepSimulator


console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="evoskill")
def main():
    """EvoSkill — Evolutionary Skill Agent.

    Skills that breed, mutate, and improve through sleep-time simulation.
    """
    pass


def _apply_provider(provider: str | None, model: str | None) -> None:
    """Apply provider/model overrides to environment."""
    if provider:
        os.environ["EVOSKILL_PROVIDER"] = provider
        # Set sensible model defaults per provider
        if not model and not os.environ.get("EVOSKILL_MODEL"):
            defaults = {
                "openai": "gpt-4o",
                "anthropic": "claude-sonnet-4-20250514",
                "deepseek": "deepseek-v4-flash",
            }
            os.environ["EVOSKILL_MODEL"] = defaults.get(provider, provider)
    if model:
        os.environ["EVOSKILL_MODEL"] = model


@main.command()
@click.option("--model", default=None, help="LLM model to use.")
@click.option("--provider", default=None, help="Provider: anthropic, openai, or deepseek.")
def init(model: str | None, provider: str | None):
    """Initialize the skill population with a primordial seed skill."""
    console.print("[bold cyan]Creating primordial seed skill...[/bold cyan]")

    seed = create_seed_skill()
    save_skill(seed)

    # Register in lineage
    tree = LineageTree()
    tree.add_skill(seed)

    console.print(f"[green]Seed skill created:[/green] {seed.name} ({seed.id})")
    console.print(f"  Tools: {', '.join(seed.genome.tool_bindings)}")
    console.print(f"  Instructions: {len(seed.genome.instructions)} rules")
    console.print(f"  Stored at: {os.environ.get('EVOSKILL_HOME', '~/.evoskill')}")


@main.command()
@click.argument("task", required=False)
@click.option("--skill-id", default=None, help="Use a specific skill ID (default: best available).")
@click.option("--model", default=None, help="LLM model to use.")
@click.option("--provider", default=None, help="Provider: anthropic, openai, or deepseek.")
def run(task: str | None, skill_id: str | None, model: str | None, provider: str | None):
    """Run a task using a skill.

    If no task is given as argument, reads from stdin.
    """
    _apply_provider(provider, model)

    if not task:
        task = sys.stdin.read().strip()
    if not task:
        console.print("[red]No task provided. Pipe a task or pass it as an argument.[/red]")
        sys.exit(1)

    # Select best skill or specific one
    if skill_id:
        skill = load_skill(skill_id)
        if not skill:
            console.print(f"[red]Skill not found: {skill_id}[/red]")
            sys.exit(1)
    else:
        skills = list_skills()
        if not skills:
            console.print("[red]No skills found. Run 'evoskill init' first.[/red]")
            sys.exit(1)
        skills.sort(key=lambda s: s.fitness, reverse=True)
        skill = skills[0]

    console.print(f"[bold]Using skill:[/bold] {skill.name} (fitness: {skill.fitness:.1f})")
    console.print(f"[bold]Task:[/bold] {task[:100]}{'...' if len(task) > 100 else ''}\n")

    agent = SkillAgent(skill, model=model)
    result = agent.run(task)

    console.print(f"\n[bold cyan]Output:[/bold cyan]")
    console.print(result["output"])
    console.print(f"\n[dim]Tools used: {result['tool_calls']} | Rounds: {result['rounds']}[/dim]")


@main.command()
@click.option("--generations", "-g", default=1, help="Number of generations to evolve.")
@click.option("--population", "-p", default=10, help="Population size.")
@click.option("--children", "-c", default=6, help="Children per generation.")
@click.option("--model", default=None, help="LLM model to use.")
@click.option("--provider", default=None, help="Provider: anthropic, openai, or deepseek.")
def evolve(generations: int, population: int, children: int, model: str | None, provider: str | None):
    """Manually trigger one or more generations of evolution."""
    _apply_provider(provider, model)

    engine = EvolutionEngine(
        population_size=population,
        children_per_generation=children,
        model=model,
    )

    for gen in range(1, generations + 1):
        console.print(f"\n[bold cyan]━━━ Generation {gen}/{generations} ━━━[/bold cyan]")
        try:
            result = engine.run_generation()
        except Exception as e:
            console.print(f"[red]Generation {gen} failed:[/red] {e}")
            console.print("[yellow]Stopping evolution. Partial progress is saved.[/yellow]")
            break

        # Print summary table
        table = Table(title=f"Generation {result['generation']} Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Best Fitness", f"{result['best_fitness']:.2f}")
        table.add_row("Avg Fitness", f"{result['avg_fitness']:.2f}")
        table.add_row("Best Skill", result.get("best_skill_name", "N/A"))
        table.add_row("Children Created", str(result["children_created"]))
        console.print(table)

    # Update lineage
    tree = sync_lineage_from_disk()
    stats = tree.stats()
    console.print(f"\n[bold]Total skills:[/bold] {stats.get('total_skills', 0)}")


@main.command()
@click.option("--max-generations", "-g", default=10, help="Maximum generations to run.")
@click.option("--population", "-p", default=10, help="Population size.")
@click.option("--children", "-c", default=6, help="Children per generation.")
@click.option("--idle", default=30, help="Idle seconds before starting.")
@click.option("--model", default=None, help="LLM model to use.")
@click.option("--provider", default=None, help="Provider: anthropic, openai, or deepseek.")
def sleep(max_generations: int, population: int, children: int, idle: int, model: str | None, provider: str | None):
    """Start sleep mode — continuous evolution during idle time.

    Runs evolutionary cycles automatically. Press Ctrl+C to stop.
    """
    _apply_provider(provider, model)
    sim = SleepSimulator(
        max_generations=max_generations,
        population_size=population,
        children_per_gen=children,
        idle_seconds=idle,
        model=model,
    )

    try:
        sim.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Sleep mode interrupted. Progress saved.[/yellow]")


@main.command()
@click.option("--skill-id", default=None, help="Show lineage from a specific skill.")
@click.option("--depth", "-d", default=5, help="Max depth to display.")
def lineage(skill_id: str | None, depth: int):
    """View the skill inheritance tree."""
    tree = sync_lineage_from_disk()

    # Stats
    stats = tree.stats()
    if not stats:
        console.print("[yellow]No lineage data. Run 'evoskill init' or 'evoskill evolve' first.[/yellow]")
        return

    console.print(f"[bold]Lineage Stats:[/bold]")
    console.print(f"  Total skills: {stats['total_skills']}")
    console.print(f"  Generations: {stats['max_generation'] + 1}")
    console.print(f"  Best fitness: {stats['best_fitness']:.2f}")
    console.print(f"  Avg fitness: {stats['avg_fitness']:.2f}")
    console.print(f"  Roots: {stats['roots']}")
    console.print(f"  Mutations: {stats['mutation_distribution']}\n")

    console.print("[bold cyan]Lineage Tree:[/bold cyan]")
    console.print(tree.to_ascii_tree(root_id=skill_id, max_depth=depth))


@main.command()
@click.option("--sort", "sort_by", default="fitness", help="Sort by: fitness, generation, name.")
@click.option("--limit", "-n", default=20, help="Max skills to show.")
def list(sort_by: str, limit: int):
    """List all skills in the population."""
    skills = list_skills()
    if not skills:
        console.print("[yellow]No skills found. Run 'evoskill init' first.[/yellow]")
        return

    # Sort
    if sort_by == "fitness":
        skills.sort(key=lambda s: s.fitness, reverse=True)
    elif sort_by == "generation":
        skills.sort(key=lambda s: s.generation, reverse=True)
    elif sort_by == "name":
        skills.sort(key=lambda s: s.name)

    table = Table(title=f"Skills ({min(limit, len(skills))} of {len(skills)})")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Gen", style="yellow")
    table.add_column("Fitness", style="green")
    table.add_column("Mutation", style="magenta")
    table.add_column("Parents")

    for s in skills[:limit]:
        fit_style = "green" if s.fitness >= 6 else ("yellow" if s.fitness >= 4 else "red")
        fit_str = f"[{fit_style}]{s.fitness:.1f}[/{fit_style}]"
        table.add_row(
            s.id,
            s.name,
            str(s.generation),
            fit_str,
            s.mutation_type.value if s.mutation_type else "seed",
            ", ".join(pid[:8] for pid in s.parent_ids) if s.parent_ids else "—",
        )

    console.print(table)


@main.command()
@click.argument("skill_id")
def delete(skill_id: str):
    """Delete a skill by ID."""
    skill = load_skill(skill_id)
    if not skill:
        console.print(f"[red]Skill not found: {skill_id}[/red]")
        sys.exit(1)

    if delete_skill(skill_id):
        console.print(f"[green]Deleted:[/green] {skill.name} ({skill_id})")
    else:
        console.print(f"[red]Failed to delete: {skill_id}[/red]")


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to.")
@click.option("--port", default=8765, help="Port to listen on.")
def gateway(host: str, port: int):
    """Start a local LLM proxy gateway.

    Presents an OpenAI-compatible API at /v1/chat/completions.
    Routes to Anthropic, OpenAI, or DeepSeek based on the model name.
    """
    from .gateway import serve_gateway
    console.print(f"[bold cyan]Starting EvoSkill LLM Gateway[/bold cyan]")
    console.print(f"  Listening on: http://{host}:{port}")
    console.print(f"  Endpoint:     http://{host}:{port}/v1/chat/completions")
    console.print(f"  Provider:     auto-detected from model name")
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")
    serve_gateway(host=host, port=port)


if __name__ == "__main__":
    main()
