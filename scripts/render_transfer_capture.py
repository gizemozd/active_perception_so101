#!/usr/bin/env python3
"""Render the representative traces stored by the transfer pilot evaluator."""

import json
import sys
from pathlib import Path

from active_perception_arms.capture_evaluation import render_capture

render_capture(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
