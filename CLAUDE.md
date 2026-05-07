# CLAUDE.md — EvoSkill

## Project context

EvoSkill is a lightweight Python experimental prototype for evolutionary AI agent skills. Skills (prompts + instructions + tools) are treated as evolvable genomes. During idle/sleep periods, the system runs generational cycles: select → mutate → evaluate → prune, with LLM-as-judge fitness scoring.

Built as a standalone project, inspired by hermes-agent's self-improving skills and openclaw's skill system.

## Commands

```bash
# Install
pip install -e .

# Run tests (no API key needed)
python -m pytest tests/ -v

# Reset and evolve
rm -rf ~/.evoskill
evoskill init
evoskill evolve -g 5 -c 4 -p 10 --provider deepseek
```

## Architecture notes

- `genome.py` — `Mutator.mutate_guided()` asks LLM to improve prompts. Falls back to random mutation if LLM call fails.
- `agent.py` — `SkillAgent.run()` sandboxes all tool execution in `tempfile.TemporaryDirectory`.
- `fitness.py` — `quick_fitness()` uses `FitnessEvaluator` with LLM judge (80% weight) + structural score (20%).
- `evolution.py` — `EvolutionEngine._create_child()` weights: 55% guided, 25% random, 20% crossover.
- `lineage.py` — `LineageTree` maintains pre-built `_children` index for O(depth) queries, persisted to `~/.evoskill/lineage.json`.

## Key files

| File | Purpose |
|------|---------|
| `evoskill/cli.py` | CLI entry (7 commands) |
| `evoskill/skill.py` | Skill dataclass + create_seed_skill() |
| `evoskill/genome.py` | SkillGenome + Mutator (guided, random, crossover) |
| `evoskill/agent.py` | SkillAgent: LLM + tool loop, sandboxed |
| `evoskill/fitness.py` | FitnessEvaluator: LLM-as-judge scoring |
| `evoskill/evolution.py` | EvolutionEngine: generational cycle |
| `evoskill/lineage.py` | LineageTree: inheritance DAG |
| `evoskill/tasks.py` | 10 built-in test tasks |
| `evoskill/tools.py` | 5 built-in tools (file, shell, search) |
| `evoskill/storage.py` | JSON persistence under ~/.evoskill/ |
| `tests/test_evolution.py` | 22 unit tests |
