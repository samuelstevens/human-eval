import sys

import fire

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str,
    k: tuple[int, ...] = (1, 10),
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    results = evaluate_functional_correctness(
        sample_file, k, n_workers, timeout, problem_file
    )
    fmt = ", ".join(f"pass@{k_i}: {{:.1f}}" for k_i in k)
    print(fmt.format(*[results[f"pass@{k_i}"] * 100 for k_i in k]))


def main():
    fire.Fire(entry_point)


if __name__ == "__main__":
    sys.exit(main())
