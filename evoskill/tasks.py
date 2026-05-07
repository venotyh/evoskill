"""Test task definitions for fitness evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .genome import ALL_AVAILABLE_TOOLS


@dataclass
class EvoTask:
    """A test task used to evaluate skill fitness."""

    id: str
    name: str
    prompt: str
    category: str  # "tool_use", "reasoning", "multi_step"
    expected_tools: list[str] = field(default_factory=list)
    reference_answer_hint: str = ""  # Hint for LLM-as-judge scoring


BUILTIN_TASKS: list[EvoTask] = [
    EvoTask(
        id="file_summary",
        name="File Summary",
        prompt=(
            "In the current working directory, create a file called 'test_data.txt' with the following content:\n"
            "apple\nbanana\ncherry\ndate\nelderberry\n"
            "Then read the file back and tell me which fruit comes first alphabetically."
        ),
        category="tool_use",
        expected_tools=["write_file", "read_file"],
        reference_answer_hint="Should create file, read it, and identify 'apple' as first alphabetically.",
    ),
    EvoTask(
        id="web_research",
        name="Web Research",
        prompt=(
            "Search the web for information about 'Python decorators'. "
            "Based on the search results, write a one-paragraph summary of what Python decorators are "
            "and save it to 'decorator_summary.txt'."
        ),
        category="multi_step",
        expected_tools=["web_search", "write_file"],
        reference_answer_hint="Should search, synthesize results, and write summary file.",
    ),
    EvoTask(
        id="shell_investigation",
        name="Shell Investigation",
        prompt=(
            "Use shell commands to find out how many Python files are in the current directory tree. "
            "Report the exact count and list their names."
        ),
        category="tool_use",
        expected_tools=["shell_exec"],
        reference_answer_hint="Should use find/ls shell command to count .py files.",
    ),
    EvoTask(
        id="logical_reasoning",
        name="Logical Reasoning",
        prompt=(
            "A farmer needs to cross a river with a fox, a chicken, and a sack of grain. "
            "The boat can only carry the farmer and one item at a time. "
            "If left alone, the fox will eat the chicken, and the chicken will eat the grain. "
            "How does the farmer get everything across safely? Explain step by step."
        ),
        category="reasoning",
        expected_tools=[],
        reference_answer_hint=(
            "Should take chicken first, return alone, take fox, bring chicken back, "
            "take grain, return alone, take chicken."
        ),
    ),
    EvoTask(
        id="code_explain",
        name="Code Explanation",
        prompt=(
            "Write a Python function called 'fibonacci' to a file 'fib.py' that returns the nth Fibonacci number. "
            "Then read the file back and explain how the function works line by line."
        ),
        category="multi_step",
        expected_tools=["write_file", "read_file"],
        reference_answer_hint="Should write a valid fibonacci function and explain the logic.",
    ),
    EvoTask(
        id="error_debug",
        name="Error Debugging",
        prompt=(
            "Imagine you're debugging a problem: a Python script fails with "
            "'KeyError: config' at line 42. The code at line 42 is:\n"
            "    db_url = settings['config']['database_url']\n"
            "What are the possible causes? Write your diagnosis to 'diagnosis.txt'."
        ),
        category="reasoning",
        expected_tools=["write_file"],
        reference_answer_hint=(
            "Should identify: missing 'config' key in settings dict, None value, "
            "nested dict access issue, and suggest using .get()."
        ),
    ),
    EvoTask(
        id="data_processing",
        name="Data Processing",
        prompt=(
            "Create a CSV file 'data.csv' with columns: name,score\n"
            "Add 5 rows of sample data. Then use a shell command to find the row with the highest score."
        ),
        category="multi_step",
        expected_tools=["write_file", "shell_exec"],
        reference_answer_hint="Should create CSV and use sort/awk to find max score.",
    ),
    EvoTask(
        id="search_organize",
        name="Search & Organize",
        prompt=(
            "Search for all text files (*.txt) in the current directory and its subdirectories. "
            "Write their paths and sizes to a file called 'txt_files_report.txt'."
        ),
        category="tool_use",
        expected_tools=["search_files", "shell_exec", "write_file"],
        reference_answer_hint="Should find txt files, get their sizes, and create a report.",
    ),
    EvoTask(
        id="planning_task",
        name="Planning Task",
        prompt=(
            "You need to organize a project with the following structure:\n"
            "- src/ with subdirectories: core, utils, api\n"
            "- tests/ mirroring src/\n"
            "- docs/ with a README.md\n"
            "Describe your plan step by step, then create at least the README.md with a project description."
        ),
        category="multi_step",
        expected_tools=["write_file"],
        reference_answer_hint="Should plan the structure logically and create the README.",
    ),
    EvoTask(
        id="system_analysis",
        name="System Analysis",
        prompt=(
            "Use shell commands to check the current operating system type, "
            "Python version installed (if any), and available disk space. "
            "Summarize your findings in a file called 'system_info.txt'."
        ),
        category="tool_use",
        expected_tools=["shell_exec", "write_file"],
        reference_answer_hint="Should use uname/ver, python --version, df and create summary.",
    ),
]
