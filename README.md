# EvoSkill

Evolutionary skill agent — skills that breed, mutate, and improve through sleep-time simulation.

## Concept

AI agent skills treated as evolvable genomes. During idle/sleep periods, the system runs generational evolution: select best skills, mutate their prompts/instructions/tools via LLM-guided operators, evaluate fitness on test tasks, and prune low-performers. Better skills survive and reproduce.

```
seed → mutate → evaluate → select → next generation → ...
```

## Quick Start

```bash
pip install -e .

# Initialize the primordial seed skill
evoskill init

# Run 5 generations of evolution
evoskill evolve -g 5 -c 4 -p 10 --provider deepseek

# View the inheritance tree
evoskill lineage

# List all skills sorted by fitness
evoskill list

# Execute a task using the best skill
echo "Search for Python files in this directory" | evoskill run

# Continuous sleep-mode evolution
evoskill sleep -g 20
```

## Architecture

```
evoskill/
├── cli.py              # CLI (click): init, run, evolve, sleep, lineage, list
├── simulator.py        # Sleep-mode background evolution
├── core/               # Domain models
│   ├── skill.py        # Skill data model (genome + metadata + fitness)
│   ├── genome.py       # Genome ops & mutation (guided/random/crossover)
│   └── tasks.py        # 10 built-in evaluation tasks
├── infra/              # I/O
│   ├── llm.py          # Unified LLM client (Anthropic / OpenAI / DeepSeek)
│   ├── storage.py      # JSON persistence (.evoskill/)
│   └── gateway.py      # OpenAI-compatible local HTTP proxy
├── runtime/            # Execution
│   ├── agent.py        # Agent loop (LLM + tool execution, sandboxed)
│   └── tools.py        # Built-in tools (file, shell, search)
└── evolution/          # Evolutionary algorithms
    ├── engine.py       # Evolution engine (select/mutate/evaluate/prune)
    ├── fitness.py      # LLM-as-judge fitness evaluation
    └── lineage.py      # Inheritance DAG with ASCII tree rendering
```

## Key Design

- **Skill = Genome + Metadata**: Genome is system_prompt + instructions + tool_bindings + parameters. Metadata tracks parent_ids, generation, fitness history, mutation type.
- **Guided Mutation** (55%): LLM analyzes current prompt and produces improved version. Higher selection pressure.
- **Random Mutation** (25%): Random prompt/tool/param changes. Maintains diversity.
- **Crossover** (20%): Two-parent recombination of instructions, tools, and params.
- **Fitness**: Agent runs task → LLM-as-judge scores output (80% weight) + structural scoring (20%).
- **Lineage Tree**: DAG with pre-built child index. O(depth) ancestor/descendant queries. ASCII tree visualization.
- **Sandboxed Execution**: All task tool calls run inside temp directory, auto-cleaned.

## Provider Setup

```bash
export DEEPSEEK_API_KEY=sk-...
evoskill evolve --provider deepseek

export ANTHROPIC_API_KEY=sk-ant-...
evoskill evolve --provider anthropic

export OPENAI_API_KEY=sk-...
evoskill evolve --provider openai
```
