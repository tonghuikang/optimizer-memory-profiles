#!/usr/bin/env python3
"""
Create memory profiling visualization matching PyTorch profiler format

(from repository root)
uv run python3 profiling/profiling_template.py
uv run python3 profiling/create_memory_plot.py
"""

import json
import gzip
from typing import Dict, List, Any, DefaultDict, Set
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import defaultdict

# Category mapping from PyTorch profiler
CATEGORY_COLORS: Dict[int, str] = {
    0: "#2E8B57",  # PARAMETER - Dark Green
    1: "#DAA520",  # OPTIMIZER_STATE - Golden Rod
    2: "#000000",  # INPUT - Black
    3: "#9370DB",  # TEMPORARY - Medium Purple
    4: "#FF0000",  # ACTIVATION - Red
    5: "#4169E1",  # GRADIENT - Medium Blue
    6: "#6A5ACD",  # AUTOGRAD_DETAIL - Slate Blue
    7: "#808080",  # Unknown - Grey
}

CATEGORY_NAMES: Dict[int, str] = {
    0: "PARAMETER",
    1: "OPTIMIZER_STATE",
    2: "INPUT",
    3: "TEMPORARY",
    4: "ACTIVATION",
    5: "GRADIENT",
    6: "AUTOGRAD_DETAIL",
    7: "Unknown",
}


def parse_memory_events(pathname: str) -> List[Dict[str, Any]]:
    """Parse memory events from compressed JSON file"""
    with gzip.open(f"{pathname}/profile.raw.json.gz", "rt") as f:
        raw_data: List[List[Any]] = json.load(f)

    events: List[Dict[str, Any]] = []
    for entry in raw_data:
        if len(entry) >= 4:
            timestamp, action, size, category = entry[:4]
            events.append(
                {
                    "timestamp": timestamp,
                    "action": action,
                    "size": size,
                    "category": category,
                }
            )

    return events


def create_memory_timeline(
    events: List[Dict[str, Any]],
) -> Dict[float, Dict[int, float]]:
    """Create memory timeline data for visualization"""
    # Handle pre-allocated memory (timestamp = -1)
    baseline_memory: DefaultDict[int, float] = defaultdict(float)
    timed_events: List[Dict[str, Any]] = []

    for event in events:
        if event["timestamp"] == -1:
            # Pre-allocated memory
            size_gb = event["size"] / (1024**3)
            if event["action"] == 1:  # Pre-existing allocation
                baseline_memory[event["category"]] += size_gb
        else:
            timed_events.append(event)

    if not timed_events:
        return {}

    # Convert timestamps to relative time in milliseconds
    start_time = min(e["timestamp"] for e in timed_events)

    # Create timeline with proper memory tracking
    timeline: Dict[float, Dict[int, float]] = {}
    current_memory: Dict[int, float] = dict(baseline_memory)

    # Sort events by timestamp
    sorted_events = sorted(timed_events, key=lambda x: x["timestamp"])

    for event in sorted_events:
        time_ms = (event["timestamp"] - start_time) / 1_000_000  # ns to ms
        category = event["category"]
        size_gb = event["size"] / (1024**3)  # Size can be negative for deallocations

        # All actions modify memory by the size amount (positive or negative)
        current_memory[category] = current_memory.get(category, 0) + size_gb

        # Ensure non-negative memory
        current_memory[category] = max(0, current_memory[category])

        # Store snapshot at this timestamp
        timeline[time_ms] = dict(current_memory)

    return timeline


def plot_memory_timeline(timeline: Dict[float, Dict[int, float]]) -> Figure:
    """Create stacked area plot matching PyTorch profiler style"""
    # Get sorted timestamps
    timestamps: List[float] = sorted(timeline.keys())

    # Get all categories that appear
    all_categories: Set[int] = set()
    for time_data in timeline.values():
        all_categories.update(time_data.keys())

    # Create data arrays for each category
    category_data: Dict[int, List[float]] = {}
    for cat in all_categories:
        category_data[cat] = []
        for ts in timestamps:
            value = timeline[ts].get(cat, 0)
            category_data[cat].append(max(0, value))  # Ensure non-negative

    # Calculate max memory for title
    max_memory: float = 0
    for ts in timestamps:
        total = sum(timeline[ts].values())
        max_memory = max(max_memory, total)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Stack the areas
    bottom = np.zeros(len(timestamps))

    for category in sorted(all_categories):
        if category in category_data and any(category_data[category]):
            color = CATEGORY_COLORS.get(category, "#808080")
            label = CATEGORY_NAMES.get(category, f"Category_{category}")

            ax.fill_between(
                timestamps,
                bottom,
                bottom + np.array(category_data[category]),
                color=color,
                label=label,
                alpha=0.8,
            )
            bottom += np.array(category_data[category])

    # Formatting
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Memory (GB)")
    ax.set_title(
        f"Max memory allocated: {max_memory:.2f} GiB\n"
        f"Max memory reserved: {max_memory:.2f} GiB"
    )

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

    # Grid and styling
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(timestamps) if timestamps else 1)
    ax.set_ylim(0, max_memory * 1.1 if max_memory > 0 else 1)

    plt.tight_layout()
    return fig


def generate_plot_from_file(pathname: str) -> None:
    events: List[Dict[str, Any]] = parse_memory_events(pathname)
    timeline: Dict[float, Dict[int, float]] = create_memory_timeline(events)
    fig: Figure = plot_memory_timeline(timeline)

    # Save the plot
    output_file: str = f"{pathname}/memory_profile_plot.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close instead of show to avoid blocking
    print(f"Memory profile plot saved as {output_file}")
