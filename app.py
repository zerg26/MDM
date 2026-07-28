"""Standalone Gradio entrypoint for the MDM debugging / human-in-the-loop UI.

    python app.py

Launches the interactive UI defined in ``src.cli.build_ui`` (single-row debugger,
batch CSV processor, and the explainable self-healing graph tab with
accept/override for human-in-the-loop validation).
"""
import os

from dotenv import load_dotenv

from src.cli import build_ui
from src.mdm.logging_config import configure_logging


def main() -> None:
    configure_logging()
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    build_ui().launch()


if __name__ == "__main__":
    main()
