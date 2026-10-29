"""Render a named MT10 checkpoint as demo + globally consolidated FTN mask GIFs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rendering import RenderConfig, visualization_cli

# Visualization defaults; all fields can also be overridden on the command line.
CONFIG = RenderConfig()

if __name__ == "__main__":
    visualization_cli("mt10", __file__, CONFIG)
