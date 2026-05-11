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
rm -rf .evoskill
evoskill init
evoskill evolve -g 5 -c 4 -p 10 --provider deepseek
```

## Package layout

```
evoskill/
├── cli.py            # CLI entry (7 commands)
├── simulator.py      # Sleep-mode orchestration
├── core/             # Domain models — no evoskill dependencies
│   ├── skill.py      # Skill dataclass + create_seed_skill()
│   ├── genome.py     # SkillGenome + Mutator (guided, random, crossover)
│   └── tasks.py      # EvoTask + BUILTIN_TASKS (10 tasks)
├── infra/            # I/O layer
│   ├── llm.py        # LLMClient: unified Anthropic / OpenAI / DeepSeek
│   ├── storage.py    # JSON persistence under .evoskill/
│   └── gateway.py    # OpenAI-compatible HTTP proxy
├── runtime/          # Execution layer
│   ├── agent.py      # SkillAgent: LLM + tool loop, sandboxed
│   └── tools.py      # 5 built-in tools (file, shell, search)
└── evolution/        # Evolutionary algorithms
    ├── engine.py     # EvolutionEngine: generational cycle
    ├── fitness.py    # FitnessEvaluator: LLM-as-judge scoring
    └── lineage.py    # LineageTree: inheritance DAG
```

Dependency flow (one direction only): `core` ← `infra` ← `runtime` ← `evolution` ← `cli`/`simulator`

## Architecture notes

- `core/genome.py` — `Mutator.mutate_guided()` asks LLM to improve prompts. Falls back to random mutation if LLM call fails.
- `runtime/agent.py` — `SkillAgent.run()` sandboxes all tool execution in `tempfile.TemporaryDirectory`.
- `evolution/fitness.py` — `quick_fitness()` uses `FitnessEvaluator` with LLM judge (80% weight) + structural score (20%).
- `evolution/engine.py` — `EvolutionEngine._create_child()` weights: 55% guided, 25% random, 20% crossover.
- `evolution/lineage.py` — `LineageTree` maintains pre-built `_children` index for O(depth) queries, persisted to `.evoskill/lineage.json`.
