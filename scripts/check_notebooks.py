"""Execute both notebooks in fresh kernels; keep verified outputs in the files."""

import argparse
from pathlib import Path
import subprocess
import sys

import nbformat
from nbclient import NotebookClient


def execute_in_process(path):
    """Socket-free alternative: fresh Python process, standard IPython execution."""
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from IPython.utils.capture import capture_output
    shell = TerminalInteractiveShell.instance()
    shell.run_line_magic("matplotlib", "inline")
    shell.display_formatter.active_types = ["text/plain", "text/html", "image/png", "image/svg+xml"]
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f"{path.name}, code cell {count}: "
                               f"{result.error_before_exec or result.error_in_exec}")
        cell.execution_count = count
        cell.outputs = []
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data,
                                                      metadata=output.metadata))
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--in-process", action="store_true", help="Execute with IPython without kernel sockets")
    parser.add_argument("--in-process-file", choices=["bigram.ipynb", "mlp.ipynb"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.in_process_file:
        sys.path.insert(0, str(root))
        execute_in_process(root / args.in_process_file)
        return
    for name in ("bigram.ipynb", "mlp.ipynb"):
        if args.in_process:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--in-process-file", name],
                           cwd=root, check=True)
            print(f"Executed {name} successfully (fresh IPython process)", flush=True)
            continue
        path = root / name
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        NotebookClient(notebook, timeout=300, kernel_name=args.kernel,
                       resources={"metadata": {"path": str(root)}}).execute()
        nbformat.write(notebook, path)
        print(f"Executed {name} successfully", flush=True)


if __name__ == "__main__":
    main()
