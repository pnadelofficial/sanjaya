"""
sanjaya CLI — generate an annotated reading environment from a TEI XML source.

Usage:
    sanjaya --config config.yaml
    sanjaya --config config.yaml --chunk 1.1 1.2
"""

import argparse
import importlib
from pathlib import Path

import yaml
from perseus_cts.models import TEIDocument

from sanjaya.site.generator import Generator


DEFAULT_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_class(dotted_path: str):
    """Import a class from a dotted module path, e.g. 'sanjaya.llm.annotators.GlossAnnotator'."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _build_annotators(annotator_configs: list) -> list:
    annotators = []
    for entry in annotator_configs:
        cls = _load_class(entry["class"])
        annotators.append(cls(**entry.get("args", {})))
    return annotators


def main():
    parser = argparse.ArgumentParser(
        prog="sanjaya",
        description="Generate an annotated reading environment from a TEI XML source.",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="FILE",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--chunk",
        nargs="*",
        metavar="ID",
        help="Process only these chunk IDs (overrides chunk_filter in config).",
    )
    cli_args = parser.parse_args()

    config_path = Path(cli_args.config).resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve relative paths against the config file's directory so configs
    # are portable — moving the config and its data together keeps things working.
    config_dir = config_path.parent

    source_file = config_dir / config["source"]["file"]
    output_dir = config_dir / config["output"]["dir"]

    raw_template_dir = config.get("templates", {}).get("dir")
    if raw_template_dir:
        template_dir = Path(raw_template_dir)
        if not template_dir.is_absolute():
            template_dir = config_dir / template_dir
    else:
        template_dir = DEFAULT_TEMPLATES_DIR

    # CLI --chunk takes precedence over config chunk_filter when provided.
    chunk_filter = cli_args.chunk or config.get("chunk_filter")

    doc = TEIDocument(str(source_file))
    annotators = _build_annotators(config["annotators"])

    gen = Generator(
        document=doc,
        template_dir=template_dir,
        subunit_xpath=config["source"]["xpath"],
        annotator_list=annotators,
        work=config["work"]["title"],
        author=config["work"]["author"],
        output_dir=output_dir,
        chunk_filter=chunk_filter,
    )
    gen.generate_site()


if __name__ == "__main__":
    main()
