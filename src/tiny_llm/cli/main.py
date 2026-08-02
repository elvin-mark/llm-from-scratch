import click
from rich.console import Console

from tiny_llm.cli.bench import bench_cmd
from tiny_llm.cli.eval import eval_cmd
from tiny_llm.cli.export import export_cmd
from tiny_llm.cli.generate import generate_cmd, infer_cmd
from tiny_llm.cli.info import info_cmd
from tiny_llm.cli.prepare_data import prepare_data_cmd
from tiny_llm.cli.train import train_cmd
from tiny_llm.cli.viz_attn import viz_attn_cmd

console = Console()


@click.group(invoke_without_command=False)
@click.version_option(version="0.1.0", prog_name="tiny-llm")
def main():
    """
    🚀 [bold cyan]TinyLLM CLI Tool[/bold cyan]

    Train, Infer, Evaluate Perplexity, Prepare Datasets, Export Weights, Benchmark, Inspect Checkpoints, and Visualize Attention Maps.
    """
    pass


main.add_command(train_cmd)
main.add_command(generate_cmd)
main.add_command(infer_cmd)
main.add_command(eval_cmd)
main.add_command(prepare_data_cmd)
main.add_command(export_cmd)
main.add_command(bench_cmd)
main.add_command(viz_attn_cmd)
main.add_command(info_cmd)

if __name__ == "__main__":
    main()
