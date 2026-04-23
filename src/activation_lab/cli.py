from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .generation import capture_reference_prefill, run_generation
from .heatmap import HeatmapRequest, generate as generate_heatmaps
from .models import load_model
from .scenario import ModelConfig, Scenario, load_scenario
from .serialize import (
    make_run_dir,
    write_conversation_snapshot,
    write_conversation_snapshot_index,
    write_reference_index,
    write_reference_state,
    write_run_manifest,
    write_step,
    write_steps_json,
)

app = typer.Typer(add_completion=False, help="Capture per-step LLM activations.")
console = Console()


def _capture_conversation_snapshots(
    scenario: Scenario, run_dir, model, tokenizer, arch, device
) -> None:
    """Capture a prefill snapshot for each message prefix in the conversation."""
    messages = scenario.prompt.messages
    if not messages:
        return
    console.print(f"  [bold]Capturing {len(messages)} conversation snapshot(s)...[/bold]")
    snap_info = []
    for i, msg in enumerate(messages):
        prefix = messages[: i + 1]
        state = capture_reference_prefill(
            model, tokenizer, arch, prefix, scenario.capture, device
        )
        write_conversation_snapshot(run_dir, i, msg.role, state)
        snap_info.append({
            "index": i,
            "role": msg.role,
            "content_preview": msg.content[:80],
        })
    write_conversation_snapshot_index(run_dir, snap_info)
    console.print(f"  [green]{len(messages)} snapshot(s) saved[/green]")


def _capture_reference_states(
    scenario: Scenario, run_dir, model, tokenizer, arch, device
) -> None:
    if not scenario.reference_states:
        return
    console.print(f"  [bold]Capturing {len(scenario.reference_states)} reference state(s)...[/bold]")
    labels: list[str] = []
    for ref in scenario.reference_states:
        console.print(f"    [dim]reference '{ref.label}'[/dim]")
        state = capture_reference_prefill(
            model, tokenizer, arch, ref.messages, scenario.capture, device
        )
        write_reference_state(run_dir, ref.label, state)
        labels.append(ref.label)
    write_reference_index(run_dir, labels)
    console.print(f"  [green]{len(labels)} reference(s) saved[/green]")


def _execute_run(scenario: Scenario, model, tokenizer, arch, device, label: str = "") -> None:
    from .generation import _build_input_ids  # local import to avoid cycle in type checkers

    if label:
        console.print(f"\n[bold]--- {label} ({len(scenario.prompt.messages)} message(s)) ---[/bold]")

    paths = make_run_dir(scenario, label=label)
    console.print(f"[bold]Run dir:[/bold] {paths.root}")

    prompt_ids = _build_input_ids(tokenizer, scenario.prompt, device)[0].tolist()
    write_run_manifest(paths, scenario, arch, tokenizer, device, prompt_ids)

    entries = []
    generated_tokens: list[str] = []
    for rec in run_generation(
        model=model,
        tokenizer=tokenizer,
        arch=arch,
        prompt_cfg=scenario.prompt,
        gen_cfg=scenario.generation,
        cap_cfg=scenario.capture,
        device=device,
    ):
        entry = write_step(paths, rec)
        entries.append(entry)
        if rec.generated_token is not None:
            generated_tokens.append(rec.generated_token)
        console.print(
            f"  step {rec.step:>3} {rec.kind:<7} seq_len={rec.seq_len:>4} "
            f"→ {rec.generated_token!r}"
        )

    write_steps_json(paths, entries)
    console.print(f"[green]Done.[/green] {len(entries)} step(s) written.")
    console.print(f"[bold]Generated:[/bold] {''.join(generated_tokens)!r}")
    _capture_conversation_snapshots(scenario, paths.root, model, tokenizer, arch, device)
    _capture_reference_states(scenario, paths.root, model, tokenizer, arch, device)


@app.command()
def run(
    scenario_path: Path = typer.Argument(..., help="Scenario YAML file or directory of YAML files."),
) -> None:
    """Execute a scenario YAML file (or every YAML in a folder) and write run directories."""
    if not scenario_path.exists():
        console.print(f"[red]Path not found: {scenario_path}[/red]")
        raise typer.Exit(1)

    if scenario_path.is_dir():
        yaml_files = sorted(scenario_path.glob("*.yaml"))
        if not yaml_files:
            console.print(f"[red]No .yaml files found in {scenario_path}[/red]")
            raise typer.Exit(1)
        console.print(f"[bold]Found {len(yaml_files)} scenario(s) in {scenario_path}[/bold]")
    else:
        yaml_files = [scenario_path]

    model = tokenizer = arch = device = None
    loaded_model_cfg = None

    for yaml_file in yaml_files:
        scenario = load_scenario(yaml_file)
        console.print(f"\n[bold]Scenario:[/bold] {scenario.name}  [dim]({yaml_file.name})[/dim]")
        console.print(f"[bold]Model:[/bold] {scenario.model.id}")

        if loaded_model_cfg != scenario.model:
            model, tokenizer, arch, device = load_model(scenario.model)
            loaded_model_cfg = scenario.model
            console.print(
                f"[bold]Loaded:[/bold] {arch.num_layers} layers, hidden={arch.hidden_size}, "
                f"device={device}, dtype={next(model.parameters()).dtype}"
            )

        if scenario.prompt.run_at_each_message:
            n = len(scenario.prompt.messages)
            console.print(f"[bold]run_at_each_message:[/bold] {n} run(s)")
            for i in range(1, n + 1):
                sub_prompt = scenario.prompt.model_copy(
                    update={"messages": scenario.prompt.messages[:i], "run_at_each_message": False}
                )
                sub_scenario = scenario.model_copy(update={"prompt": sub_prompt})
                _execute_run(sub_scenario, model, tokenizer, arch, device, label=f"msg{i}of{n}")
        else:
            _execute_run(scenario, model, tokenizer, arch, device)


@app.command()
def layers(
    model_id: str = typer.Argument(..., help="HF model id, e.g. Qwen/Qwen2.5-3B-Instruct"),
    trust_remote_code: bool = typer.Option(False),
) -> None:
    """Print the module tree of a model — useful for designing hook targets."""
    model, _tok, arch, _device = load_model(
        ModelConfig(id=model_id, device="cpu", dtype="float32", trust_remote_code=trust_remote_code)
    )
    console.print(
        f"num_layers={arch.num_layers}  hidden={arch.hidden_size}  "
        f"heads={arch.num_attention_heads}  kv_heads={arch.num_key_value_heads}"
    )
    for name, _mod in model.named_modules():
        depth = name.count(".")
        if depth <= 3:
            console.print(f"  {'  ' * depth}{name}")


@app.command()
def inspect(
    run_json_path: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Summarize a completed run (run.json + steps.json)."""
    run = json.loads(run_json_path.read_text())
    steps_path = run_json_path.parent / "steps.json"
    steps = json.loads(steps_path.read_text())["steps"] if steps_path.exists() else []

    console.print(f"[bold]Scenario:[/bold] {run['scenario']['name']}")
    console.print(f"[bold]Model:[/bold] {run['model']['id']}")
    console.print(f"[bold]Device:[/bold] {run['env']['device']}")
    console.print(f"[bold]Steps:[/bold] {len(steps)}")

    if steps:
        table = Table(title="Step summary")
        table.add_column("step")
        table.add_column("kind")
        table.add_column("seq_len")
        table.add_column("gen_token")
        table.add_column("logprob")
        table.add_column("tensors")
        for s in steps:
            table.add_row(
                str(s["step"]),
                s["kind"],
                str(s["seq_len"]),
                repr(s["generated_token"]),
                f"{s['logit_argmax_logprob']:.3f}" if s["logit_argmax_logprob"] is not None else "-",
                str(len(s["tensor_index"])),
            )
        console.print(table)


def _parse_int_list(spec: str | None) -> list[int] | None:
    if spec is None or spec.lower() == "all":
        return None
    return [int(x) for x in spec.split(",") if x.strip()]


@app.command()
def heatmap(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    source: str = typer.Option(
        "hidden_out",
        help="Tensor key to plot (hidden_out, mlp_act, q, k, v, attn_weights, ...).",
    ),
    layers: str = typer.Option("all", help="Comma-separated layer indices or 'all'."),
    steps: str = typer.Option("all", help="Comma-separated step indices or 'all'."),
    per_head: bool = typer.Option(
        False, "--per-head", help="For attn_weights, emit one image per head instead of head-mean."
    ),
    normalize: str = typer.Option(
        "per_image",
        help="Color scaling: none | per_image | signed | global.",
    ),
    reduce: str = typer.Option(
        "signed",
        help="Value mapping before plotting: signed (keep sign) or abs (magnitude).",
    ),
    cmap: str = typer.Option("viridis", help="Matplotlib colormap name."),
) -> None:
    """Render per-(layer, step) activation heatmaps for a completed run."""
    req = HeatmapRequest(
        run_dir=run_dir,
        source=source,
        layers=_parse_int_list(layers),
        steps=_parse_int_list(steps),
        per_head=per_head,
        normalize=normalize,  # type: ignore[arg-type]
        reduce=reduce,  # type: ignore[arg-type]
        cmap=cmap,
    )
    console.print(f"[bold]Heatmap:[/bold] {run_dir} source={source}")
    written = generate_heatmaps(req)
    console.print(f"[green]Wrote {len(written)} image(s)[/green] to {run_dir}/heatmaps/{source}/")


@app.command("logit-lens")
def logit_lens_cmd(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    steps: str = typer.Option("all", help="Comma-separated step indices or 'all'."),
    position: int = typer.Option(-1, help="Token position within each step's tensor (-1 = last)."),
    top_k: int = typer.Option(10, help="Top-k tokens to record per layer."),
) -> None:
    """Project every layer's hidden_out through final_norm + lm_head and save per-step top-k tokens."""
    from dataclasses import asdict

    from .view import Run

    run = Run(run_dir)
    model, tok = run.load_model()

    step_ids = (
        [s.step for s in run.steps] if steps.lower() == "all"
        else [int(x) for x in steps.split(",") if x.strip()]
    )

    console.print(f"[bold]Logit lens:[/bold] {len(step_ids)} step(s), {run.num_layers} layer(s)")
    out: dict[str, object] = {"model": run.model_id, "position": position, "top_k": top_k, "frames": []}
    frames_all = []
    for sid in step_ids:
        frames = run.logit_lens(sid, position=position, top_k=top_k)
        frames_all.extend(frames)
        console.print(
            f"  step {sid:>3}: final argmax = {frames[-1].top_k[0]['token']!r} "
            f"({frames[-1].argmax_logprob:.3f})"
        )
    out["frames"] = [asdict(f) for f in frames_all]

    dst = run.path / "logit_lens.json"
    dst.write_text(json.dumps(out, indent=2, default=str))
    console.print(f"[green]Wrote[/green] {dst}")


@app.command()
def view(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    port: int = typer.Option(8501, help="Port for the Streamlit server."),
) -> None:
    """Launch the Streamlit UI for a run directory."""
    import os
    import subprocess
    from pathlib import Path as _P

    app_path = _P(__file__).parent / "view" / "app.py"
    env = os.environ.copy()
    env["ACTIVATION_LAB_RUN"] = str(run_dir.resolve())
    console.print(f"[bold]Launching UI:[/bold] http://localhost:{port}")
    console.print(f"[dim]run: {run_dir}[/dim]")
    subprocess.run(
        [
            "streamlit", "run", str(app_path),
            "--server.port", str(port),
            "--server.headless", "true",
        ],
        env=env,
        check=False,
    )


if __name__ == "__main__":
    app()
