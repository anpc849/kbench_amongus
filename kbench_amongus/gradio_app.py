from __future__ import annotations

import copy
import argparse
import html
import importlib
import json
import math
import os
import queue
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

try:
    from faker import Faker
except Exception:  # pragma: no cover - optional runtime dependency fallback
    Faker = None

from kbench_amongus import ConfiguredAmongUs, DefaultLLMAgent, GameConfig
from kbench_amongus.envs.configs.map_config import (
    connections as DEFAULT_CONNECTIONS,
    room_data as DEFAULT_ROOM_DATA,
    vent_connections as DEFAULT_VENT_CONNECTIONS,
)
from kbench_amongus.envs.configs.task_config import task_config as DEFAULT_TASK_CONFIG


MAX_PLAYERS = 8
MAX_ROOMS = 18
MAX_EDGES = 28
MAX_TASKS = 32
TASK_TYPES = ["common", "short", "long"]
DEFAULT_MODEL_IDS = [
    "anthropic/claude-opus-4-8@default",
    "openai/gpt-5.5-2026-04-23",
    "qwen/qwen3-235b-a22b-instruct-2507",
    "qwen/qwen3-235b-a22b-instruct-2507",
    "google/gemini-3-flash-preview",
]
DEFAULT_PERSONALITIES = [
    "Careful impostor: build a believable route, avoid overclaiming, and adapt to direct evidence.",
    "Task-first crewmate: prioritize efficient task routes, but pause for strong direct evidence.",
    "Evidence keeper: remember who was where and ask precise timing questions in meetings.",
    "Buddy-system crewmate: avoid isolation and compare group movement during meetings.",
    "Skeptical crewmate: challenge contradictions and separate direct evidence from suspicion.",
]
SAFE_PLAYER_COLORS = [
    "#f59f00",
    "#ff6b6b",
    "#4dabf7",
    "#51cf66",
    "#cc5de8",
    "#20c997",
    "#ff922b",
    "#748ffc",
]
FALLBACK_PLAYER_NAMES = [
    "Alex",
    "Blair",
    "Casey",
    "Drew",
    "Emery",
    "Finley",
    "Gray",
    "Harper",
    "Indigo",
    "Jordan",
    "Kai",
    "Logan",
]
NAME_FAKER = Faker() if Faker is not None else None
DEFAULT_SKELD_POSITIONS = {
    "Reactor": (95, 350),
    "Upper Engine": (205, 155),
    "Lower Engine": (205, 500),
    "Security": (315, 365),
    "Medbay": (365, 230),
    "Electrical": (390, 445),
    "Storage": (500, 500),
    "Cafeteria": (520, 150),
    "Admin": (625, 365),
    "O2": (675, 270),
    "Weapons": (740, 150),
    "Navigation": (850, 315),
    "Shields": (790, 495),
    "Communications": (640, 575),
}
RUN_STOP_EVENTS: dict[str, threading.Event] = {}


class GameStopped(Exception):
    """Raised by the Gradio observer when the user stops a running game."""


@dataclass
class GradioSnapshot:
    timestep: int
    phase: str
    task_progress: float
    map_payload: dict[str, Any]
    players: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    activities: list[dict[str, Any]]
    meeting_info: dict[str, Any]
    report_text: str = ""


class PersonalityLLMAgent(DefaultLLMAgent):
    """Default kbench LLM agent with a user-authored personality note."""

    def __init__(
        self,
        llm,
        *,
        personality: str = "",
        max_retries: int = 2,
        llm_pause_seconds: float = 1.0,
    ):
        super().__init__(
            llm,
            max_retries=max_retries,
            llm_pause_seconds=llm_pause_seconds,
        )
        self.personality = personality.strip()

    def _system_text(self, context) -> str:
        text = super()._system_text(context)
        if not self.personality:
            return text
        return (
            f"{text}\n\n"
            "Player personality / decision style:\n"
            f"{self.personality}\n"
        )

    def agent_name(self) -> str:
        return "PersonalityLLMAgent"

    def get_log(self) -> dict[str, Any] | None:
        payload = super().get_log() or {}
        payload["personality"] = self.personality
        return payload


class GradioGameUI:
    """Observer used by the runner to stream snapshots to Gradio."""

    def __init__(
        self,
        updates: "queue.Queue[GradioSnapshot]",
        stop_event: threading.Event | None = None,
    ):
        self.updates = updates
        self.stop_event = stop_event
        self.last_env = None
        self.report_text = ""

    def reset(self):
        self.last_env = None
        self.report_text = ""

    def draw_map(self, env):
        if self.stop_event is not None and self.stop_event.is_set():
            raise GameStopped("Game stopped by user.")
        self.last_env = env
        self.updates.put(snapshot_from_env(env, self.report_text))

    def report(self, text):
        if self.stop_event is not None and self.stop_event.is_set():
            raise GameStopped("Game stopped by user.")
        self.report_text = text
        if self.last_env is not None:
            self.updates.put(snapshot_from_env(self.last_env, self.report_text))

    def quit_UI(self):
        return None


def load_kbench():
    import kaggle_benchmarks as kbench

    try:
        if len(list(kbench.llms.keys())) > 0:
            return kbench
    except Exception as exc:
        raise RuntimeError("Unable to inspect kbench.llms.") from exc

    root = Path.cwd()
    local_src = root / "kaggle-benchmarks" / "src"
    env_file = root / "kaggle-benchmarks" / ".env"
    if local_src.exists() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))
    if env_file.exists():
        load_env_file(env_file)

    kbench = importlib.reload(kbench)
    try:
        if len(list(kbench.llms.keys())) > 0:
            return kbench
    except Exception as exc:
        raise RuntimeError("Unable to inspect kbench.llms after loading .env.") from exc
    raise RuntimeError("kbench.llms is empty after loading local .env.")


def load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def model_choices(kbench) -> list[str]:
    choices = list(getattr(kbench, "llms", {}).keys())
    if choices:
        return choices
    return list(DEFAULT_MODEL_IDS)


def default_rooms() -> list[dict[str, Any]]:
    return [{"enabled": True, "name": room} for room in DEFAULT_ROOM_DATA]


def default_edges(edges) -> list[dict[str, Any]]:
    return [
        {"enabled": True, "from": room_a, "to": room_b}
        for room_a, room_b in edges
    ]


def default_tasks() -> list[dict[str, Any]]:
    rows = []
    for room, details in DEFAULT_ROOM_DATA.items():
        for task_name in details["tasks"]:
            config = DEFAULT_TASK_CONFIG.get(task_name)
            if config:
                rows.append(
                    {
                        "enabled": True,
                        "name": task_name,
                        "room": room,
                        "type": config["task_type"],
                        "duration": int(config["duration"]),
                    }
                )
    return rows


def random_player_alias(index: int) -> str:
    if NAME_FAKER is not None:
        return NAME_FAKER.first_name()
    return random.choice(FALLBACK_PLAYER_NAMES)


def default_players(choices: list[str]) -> list[dict[str, Any]]:
    players = []
    for index in range(5):
        model = DEFAULT_MODEL_IDS[index]
        if model not in choices and choices:
            model = choices[min(index, len(choices) - 1)]
        players.append(
            {
                "enabled": True,
                "name": random_player_alias(index),
                "role": "impostor" if index == 0 else "crewmate",
                "model": model,
                "personality": "",
            }
        )
    return players


def random_player_colors(count: int) -> list[str]:
    colors = list(SAFE_PLAYER_COLORS)
    random.SystemRandom().shuffle(colors)
    return colors[:count]


def build_room_data(
    rooms: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    vent_edges: list[tuple[str, str]],
) -> dict[str, dict[str, Any]]:
    room_names = [row["name"] for row in rooms if row.get("enabled")]
    room_data = {
        room_name: {
            "tasks": [],
            "vent": [],
            "special_actions": ["Emergency Button"] if room_name == "Cafeteria" else [],
            "players": [],
        }
        for room_name in room_names
    }
    for task in tasks:
        if task.get("enabled"):
            room_data[task["room"]]["tasks"].append(task["name"])
    for room_a, room_b in vent_edges:
        room_data[room_a]["vent"].append(room_b)
        room_data[room_b]["vent"].append(room_a)
    return room_data


def dedupe_edges(edges: list[dict[str, Any]]) -> list[tuple[str, str]]:
    seen = set()
    cleaned = []
    for row in edges:
        if not row.get("enabled"):
            continue
        room_a = str(row.get("from", "")).strip()
        room_b = str(row.get("to", "")).strip()
        if not room_a or not room_b or room_a == room_b:
            continue
        key = tuple(sorted((room_a, room_b)))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((room_a, room_b))
    return cleaned


def validate_config_payload(
    players: list[dict[str, Any]],
    rooms: list[dict[str, Any]],
    corridors: list[dict[str, Any]],
    vents: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    settings: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[tuple[str, str]], list[tuple[str, str]], dict[str, dict[str, Any]]]:
    errors = []
    active_players = [player for player in players if player.get("enabled")]
    if len(active_players) < 2:
        errors.append("Enable at least two players.")
    if sum(1 for player in active_players if player["role"] == "impostor") < 1:
        errors.append("Enable at least one impostor.")
    if any(not player.get("model") for player in active_players):
        errors.append("Every enabled player must have a model.")

    active_rooms = [room for room in rooms if room.get("enabled") and room.get("name")]
    room_names = [str(room["name"]).strip() for room in active_rooms]
    if len(room_names) != len(set(room_names)):
        errors.append("Room names must be unique.")
    if "Cafeteria" not in set(room_names):
        errors.append("A room named Cafeteria is required as the start/emergency room.")

    room_set = set(room_names)
    corridor_edges = dedupe_edges(corridors)
    vent_edges = dedupe_edges(vents)
    for edge_type, edges in (("corridor", corridor_edges), ("vent", vent_edges)):
        for room_a, room_b in edges:
            if room_a not in room_set or room_b not in room_set:
                errors.append(
                    f"{edge_type} edge {room_a} <-> {room_b} references a missing room."
                )

    active_tasks = [task for task in tasks if task.get("enabled")]
    task_config = {}
    task_seen = set()
    for task in active_tasks:
        task_name = str(task.get("name", "")).strip()
        room = str(task.get("room", "")).strip()
        task_type = str(task.get("type", "")).strip()
        try:
            duration = int(task.get("duration", 1))
        except (TypeError, ValueError):
            duration = 0
        if not task_name:
            errors.append("Every enabled task must have a name.")
            continue
        if room not in room_set:
            errors.append(f"Task {task_name!r} references missing room {room!r}.")
        if task_type not in TASK_TYPES:
            errors.append(f"Task {task_name!r} has invalid type {task_type!r}.")
        if duration < 1:
            errors.append(f"Task {task_name!r} duration must be at least 1.")
        task_key = (task_name, room)
        if task_key in task_seen:
            errors.append(f"Duplicate task {task_name!r} in room {room!r}.")
        task_seen.add(task_key)
        task_config[task_name] = {"duration": duration, "task_type": task_type}

    graph = nx.Graph()
    graph.add_nodes_from(room_names)
    graph.add_edges_from(corridor_edges)
    if "Cafeteria" in graph:
        unreachable = sorted(node for node in graph.nodes if not nx.has_path(graph, "Cafeteria", node))
        if unreachable:
            errors.append(
                "Every enabled room must be reachable from Cafeteria by corridors. "
                f"Unreachable: {', '.join(unreachable)}."
            )
    if graph.number_of_edges() == 0:
        errors.append("Add at least one corridor.")

    crewmate_count = sum(1 for player in active_players if player["role"] == "crewmate")
    type_counts = {
        task_type: sum(1 for task in active_tasks if task.get("type") == task_type)
        for task_type in TASK_TYPES
    }
    if int(settings["num_common_tasks"]) > type_counts["common"]:
        errors.append("num_common_tasks exceeds available enabled common tasks.")
    if int(settings["num_short_tasks"]) * crewmate_count > type_counts["short"]:
        errors.append("num_short_tasks * crewmates exceeds available enabled short tasks.")
    if int(settings["num_long_tasks"]) * crewmate_count > type_counts["long"]:
        errors.append("num_long_tasks * crewmates exceeds available enabled long tasks.")

    if errors:
        raise ValueError("\n".join(f"- {error}" for error in errors))

    room_data = build_room_data(active_rooms, active_tasks, vent_edges)
    return active_players, room_data, corridor_edges, vent_edges, task_config


def make_game_config(kbench, state: dict[str, Any]) -> GameConfig:
    settings = state["settings"]
    players, room_data, corridors, vents, task_config = validate_config_payload(
        state["players"],
        state["rooms"],
        state["corridors"],
        state["vents"],
        state["tasks"],
        settings,
    )
    player_configs = []
    colors = random_player_colors(len(players))
    for index, player in enumerate(players):
        llm = kbench.llms[player["model"]]
        agent = PersonalityLLMAgent(llm, personality=player.get("personality", ""))
        player_configs.append(
            {
                "role": player["role"],
                "agent": agent,
                "name": player["name"],
                "location": "Cafeteria",
                "color": colors[index],
                "model_id": player["model"],
            }
        )
    game_settings = {
        "num_common_tasks": int(settings["num_common_tasks"]),
        "num_short_tasks": int(settings["num_short_tasks"]),
        "num_long_tasks": int(settings["num_long_tasks"]),
        "discussion_rounds": int(settings["discussion_rounds"]),
        "max_num_buttons": int(settings["max_num_buttons"]),
        "kill_cooldown": int(settings["kill_cooldown"]),
        "max_timesteps": int(settings["max_timesteps"]),
    }
    return GameConfig(
        game_settings=game_settings,
        player_configs=player_configs,
        task_config=task_config,
        map_room_data=room_data,
        map_connections=corridors,
        map_vent_connections=vents,
        seed=int(settings["seed"]),
    )


def export_config_payload(state: dict[str, Any]) -> dict[str, Any]:
    players, room_data, corridors, vents, task_config = validate_config_payload(
        state["players"],
        state["rooms"],
        state["corridors"],
        state["vents"],
        state["tasks"],
        state["settings"],
    )
    settings = {
        key: int(value)
        for key, value in state["settings"].items()
        if key != "seed"
    }
    return {
        "settings": settings,
        "seed": int(state["settings"]["seed"]),
        "players": [
            {
                "role": player["role"],
                "name": player["name"],
                "model_id": player["model"],
                "personality": player.get("personality", ""),
                "location": "Cafeteria",
            }
            for player in players
        ],
        "task_config": task_config,
        "map_room_data": room_data,
        "map_connections": [list(edge) for edge in corridors],
        "map_vent_connections": [list(edge) for edge in vents],
    }


def game_config_from_export(payload: dict[str, Any], kbench) -> GameConfig:
    player_configs = []
    colors = random_player_colors(len(payload["players"]))
    for index, player in enumerate(payload["players"]):
        agent = PersonalityLLMAgent(
            kbench.llms[player["model_id"]],
            personality=player.get("personality", ""),
        )
        player_configs.append(
            {
                "role": player["role"],
                "agent": agent,
                "name": player["name"],
                "location": player.get("location", "Cafeteria"),
                "color": colors[index],
                "model_id": player["model_id"],
            }
        )
    return GameConfig(
        game_settings=dict(payload["settings"]),
        player_configs=player_configs,
        task_config={
            name: dict(config)
            for name, config in payload["task_config"].items()
        },
        map_room_data=copy.deepcopy(payload["map_room_data"]),
        map_connections=[tuple(edge) for edge in payload["map_connections"]],
        map_vent_connections=[tuple(edge) for edge in payload["map_vent_connections"]],
        seed=int(payload.get("seed", 7)),
    )


def export_config_code(state: dict[str, Any]) -> str:
    payload = export_config_payload(state)
    return (
        "import kbench_amongus as amongus\n"
        "from kbench_amongus.gradio_app import game_config_from_export\n\n"
        "game_config_payload = "
        + json.dumps(payload, indent=2)
        + "\n\n"
        "game_config = game_config_from_export(game_config_payload, kbench)\n"
        "result = amongus.run_amongus_game(game_config=game_config, UI=None)\n"
    )


def snapshot_from_env(env, report_text: str = "") -> GradioSnapshot:
    map_payload = env._map_payload() if hasattr(env, "_map_payload") else {}
    model_by_player = {
        f"{config['name']}: {config['color']}": config.get("model_id", "")
        for config in getattr(env, "resolved_player_configs", [])
    }
    meeting_info = dict(getattr(env, "last_meeting_info", None) or {})
    meeting_info["is_voting"] = (
        env.current_phase == "meeting"
        and getattr(env, "discussion_rounds_left", 1) == 0
    )
    players = []
    for index, player in enumerate(env.players, start=1):
        players.append(
            {
                "display_id": index,
                "name": player.name,
                "role": player.identity,
                "alive": player.is_alive,
                "reported_death": player.reported_death,
                "location": player.location,
                "color": player.color,
                "model": model_by_player.get(
                    player.name,
                    env.agent_metadata.get(player.name, {}).get("model", ""),
                ),
            }
        )
    activities = [
        {
            key: str(value)
            for key, value in record.items()
            if key in {"timestep", "phase", "round", "player", "action"}
        }
        for record in env.activity_log
    ]
    return GradioSnapshot(
        timestep=env.timestep,
        phase=env.current_phase,
        task_progress=env.task_assignment.check_task_completion(),
        map_payload=map_payload,
        players=players,
        decisions=list(getattr(env, "decision_log", [])),
        activities=activities,
        meeting_info=meeting_info,
        report_text=report_text,
    )


def player_prefix(player_name: str) -> str:
    return str(player_name).split(":", 1)[0].strip()


def player_number_label(player_name: str) -> str:
    prefix = player_prefix(player_name)
    match = re.search(r"Player\s+(\d+)", prefix)
    if match:
        return match.group(1)
    parts = prefix.split()
    if parts and parts[-1].isdigit():
        return parts[-1]
    return prefix[:2].upper()


def short_model_name(model: str) -> str:
    text = str(model or "").strip()
    if not text:
        return "unknown-model"
    return text.split("/")[-1]


def player_meta_model(player_meta: dict[str, Any]) -> str:
    return str(player_meta.get("model") or player_meta.get("config", {}).get("model_id", ""))


def player_lookup(players: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for player in players:
        full_name = str(player["name"])
        prefix = player_prefix(full_name)
        lookup[full_name] = player
        lookup[prefix] = player
        lookup[normalize_player_reference(full_name)] = player
        lookup[normalize_player_reference(prefix)] = player
    return lookup


def normalize_player_reference(player_name: str) -> str:
    text = str(player_name or "").strip()
    text = re.sub(r"\s+\((Crewmate|Impostor|Imposter)\)\s*$", "", text, flags=re.I)
    text = player_prefix(text)
    return text.strip().lower()


def find_player(player_name: str, players: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = player_lookup(players)
    text = str(player_name or "").strip()
    return (
        lookup.get(text)
        or lookup.get(player_prefix(text))
        or lookup.get(normalize_player_reference(text))
        or {}
    )


def player_display_name(player_name: str, players: list[dict[str, Any]]) -> str:
    player = find_player(player_name, players)
    label = player_prefix(str(player.get("name") or player_name))
    return f"{label}: {short_model_name(player.get('model', ''))}"


def player_color(player_name: str, players: list[dict[str, Any]]) -> str:
    player = find_player(player_name, players)
    return str(player.get("color") or "#f59f00")


def player_role_label(player_name: str, players: list[dict[str, Any]]) -> str:
    player = find_player(player_name, players)
    return str(player.get("role") or "unknown")


def role_pill_html(role: str) -> str:
    role_text = str(role or "unknown").strip()
    role_class = "role-impostor" if role_text.lower() == "impostor" else "role-crewmate"
    return (
        f"<span class='step-pill role-pill {role_class}'>"
        f"{html.escape(role_text.upper())}</span>"
    )


def graph_positions(map_payload: dict[str, Any]) -> dict[str, tuple[float, float]]:
    rooms = list(map_payload.get("rooms", {}))
    graph = nx.Graph()
    graph.add_nodes_from(rooms)
    graph.add_edges_from(map_payload.get("corridors", []))
    if not rooms:
        return {}
    if all(room in DEFAULT_SKELD_POSITIONS for room in rooms):
        return {room: DEFAULT_SKELD_POSITIONS[room] for room in rooms}
    try:
        raw = nx.spring_layout(graph, seed=7, k=1.75, iterations=180)
    except Exception:
        raw = {
            room: (
                math.cos(2 * math.pi * index / max(1, len(rooms))),
                math.sin(2 * math.pi * index / max(1, len(rooms))),
            )
            for index, room in enumerate(rooms)
        }
    xs = [pos[0] for pos in raw.values()]
    ys = [pos[1] for pos in raw.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale_x = max(max_x - min_x, 0.001)
    scale_y = max(max_y - min_y, 0.001)
    return {
        room: (
            80 + ((pos[0] - min_x) / scale_x) * 780,
            75 + ((pos[1] - min_y) / scale_y) * 470,
        )
        for room, pos in raw.items()
    }


def render_map(snapshot: GradioSnapshot | None, replay_index: int | None = None) -> str:
    if snapshot is None:
        return empty_map_html("Configure players and game settings, then press Play.")
    positions = graph_positions(snapshot.map_payload)
    players_by_room: dict[str, list[dict[str, Any]]] = {}
    for player in snapshot.players:
        players_by_room.setdefault(player["location"], []).append(player)
    edge_svg = []
    corridor_pairs = {
        frozenset((room_a, room_b))
        for room_a, room_b in snapshot.map_payload.get("corridors", [])
    }
    for room_a, room_b in snapshot.map_payload.get("corridors", []):
        if room_a in positions and room_b in positions:
            x1, y1 = positions[room_a]
            x2, y2 = positions[room_b]
            edge_svg.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                'class="corridor" />'
            )
    for room_a, room_b in snapshot.map_payload.get("vents", []):
        if room_a in positions and room_b in positions:
            x1, y1 = positions[room_a]
            x2, y2 = positions[room_b]
            if frozenset((room_a, room_b)) in corridor_pairs:
                dx = x2 - x1
                dy = y2 - y1
                length = max((dx * dx + dy * dy) ** 0.5, 1)
                offset = 12
                ox = -dy / length * offset
                oy = dx / length * offset
                x1 += ox
                y1 += oy
                x2 += ox
                y2 += oy
            edge_svg.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                'class="vent" />'
            )
    node_svg = []
    for room, (x, y) in positions.items():
        details = snapshot.map_payload.get("rooms", {}).get(room, {})
        task_count = len(details.get("tasks", []))
        width = max(118, min(172, 72 + len(room) * 6))
        height = 62
        node_svg.append(
            f'<rect x="{x - width / 2:.1f}" y="{y - height / 2:.1f}" '
            f'width="{width}" height="{height}" '
            'rx="8" class="room" />'
        )
        node_svg.append(
            f'<text x="{x:.1f}" y="{y - 7:.1f}" class="room-label">'
            f"{html.escape(room)}</text>"
        )
        node_svg.append(
            f'<text x="{x:.1f}" y="{y + 12:.1f}" class="task-label">'
            f"{task_count} tasks</text>"
        )
        for index, player in enumerate(players_by_room.get(room, [])):
            px = x - min(width / 2 - 16, 48) + (index % 5) * 22
            py = y + height / 2 + 15 + (index // 5) * 19
            dead_class = " dead" if not player["alive"] else ""
            node_svg.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" '
                f'fill="{html.escape(player["color"])}" class="player{dead_class}" />'
            )
            if not player["alive"]:
                node_svg.append(
                    f'<line x1="{px - 6:.1f}" y1="{py - 6:.1f}" '
                    f'x2="{px + 6:.1f}" y2="{py + 6:.1f}" class="dead-x" />'
                )
                node_svg.append(
                    f'<line x1="{px + 6:.1f}" y1="{py - 6:.1f}" '
                    f'x2="{px - 6:.1f}" y2="{py + 6:.1f}" class="dead-x" />'
                )
    replay_text = "" if replay_index is None else f"Replay step {replay_index}"
    meeting = snapshot.meeting_info
    meeting_text = ""
    if snapshot.phase == "meeting" and meeting.get("is_voting"):
        meeting_text = (
            "<div class='meeting-banner voting-banner'>"
            "<div class='meeting-icon'>V</div>"
            "<div><div class='meeting-title'>Voting phase</div>"
            f"<div class='meeting-meta'>Timestep {snapshot.timestep} | Discussion is closed. Votes are being cast.</div></div>"
            "</div>"
        )
    elif snapshot.phase == "meeting":
        newly_dead = ", ".join(
            player["name"] for player in meeting.get("newly_known_dead_players", [])
        ) or "none"
        meeting_text = (
            "<div class='meeting-banner'>"
            "<div class='meeting-icon'>!</div>"
            "<div><div class='meeting-title'>Emergency meeting</div>"
            f"<div class='meeting-meta'>Timestep {snapshot.timestep} | "
            f"{html.escape(meeting.get('trigger', 'active'))} | "
            f"Newly known dead: {html.escape(newly_dead)}</div></div>"
            "</div>"
        )
    elif any(str(item.get("action", "")).startswith("VOTE") for item in snapshot.activities[-8:]):
        meeting_text = "<div class='phase voting'>Voting in progress</div>"
    else:
        meeting_text = "<div class='phase task'>Task phase</div>"
    report = render_outcome_banner(snapshot.report_text)
    svg = (
        '<svg viewBox="0 0 940 620" class="map-svg" role="img">'
        f"{''.join(edge_svg)}{''.join(node_svg)}</svg>"
    )
    if snapshot.phase == "meeting":
        stage = render_meeting_conversation(snapshot, full=True)
    else:
        stage = svg
    progress = max(0, min(100, snapshot.task_progress * 100))
    return f"""
<div class="game-shell">
  <div class="topbar">
    <div><strong>Step {snapshot.timestep}</strong> | {html.escape(snapshot.phase)} | {replay_text}</div>
    <div class="progress-wrap" aria-label="Task progress">
      <span>Task progress</span>
      <div class="progress-track"><div class="progress-fill" style="width:{progress:.1f}%"></div></div>
      <strong>{progress:.1f}%</strong>
    </div>
  </div>
  {report}
  {meeting_text}
  {stage}
</div>
"""


def render_outcome_banner(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    icon = "!"
    outcome_class = "neutral"
    title = "Game ended"
    if "crewmates win" in lowered:
        icon = "C"
        outcome_class = "crew"
        title = "Crewmates win"
    elif "impostors win" in lowered:
        icon = "I"
        outcome_class = "impostor"
        title = "Impostors win"
    elif "stopped" in lowered or "stop requested" in lowered:
        icon = "S"
        outcome_class = "neutral"
        title = "Stopped"
    return (
        f"<div class='outcome-banner {outcome_class}'>"
        f"<div class='outcome-icon'>{icon}</div>"
        "<div>"
        f"<div class='outcome-title'>{html.escape(title)}</div>"
        f"<div class='outcome-message'>{html.escape(text)}</div>"
        "</div>"
        "</div>"
    )


def empty_map_html(message: str) -> str:
    return f"<div class='empty-map'><div>{html.escape(message)}</div></div>"


def render_meeting_conversation(snapshot: GradioSnapshot, full: bool = False) -> str:
    bubbles = []
    votes: list[tuple[str, str]] = []
    for activity in snapshot.activities:
        action = str(activity.get("action", ""))
        speaker_raw = str(activity.get("player", ""))
        color = player_color(speaker_raw, snapshot.players)
        label = player_display_name(speaker_raw, snapshot.players)
        if "SPEAK:" in action:
            message = action.split("SPEAK:", 1)[1].strip()
            bubbles.append(
                "<div class='meeting-bubble'>"
                f"<div class='meeting-speaker-dot' style='background:{html.escape(str(color))}'></div>"
                "<div>"
                f"<div class='meeting-speaker'>{html.escape(label)}</div>"
                f"<div class='meeting-message'>{html.escape(message)}</div>"
                "</div>"
                "</div>"
            )
            continue
        if action.startswith("VOTE "):
            target_raw = action.split("VOTE ", 1)[1].strip()
            target = player_display_name(target_raw, snapshot.players)
            votes.append((label, target))
            bubbles.append(
                "<div class='meeting-bubble vote-bubble'>"
                f"<div class='meeting-speaker-dot' style='background:{html.escape(str(color))}'></div>"
                "<div>"
                f"<div class='meeting-speaker'>{html.escape(label)}</div>"
                f"<div class='meeting-message'><span class='vote-chip'>VOTE</span> "
                f"{html.escape(target)}</div>"
                "</div>"
                "</div>"
            )
    if not bubbles:
        bubbles.append("<div class='activity'>No meeting messages yet.</div>")
    vote_summary = render_vote_summary(votes)
    full_class = " full" if full else ""
    return (
        f"<div class='meeting-conversation{full_class}'>"
        "<div class='meeting-conversation-title'>Meeting conversation</div>"
        + vote_summary
        + "".join(bubbles)
        + "</div>"
    )


def render_vote_summary(votes: list[tuple[str, str]]) -> str:
    if not votes:
        return ""
    counts: dict[str, int] = {}
    for _voter, target in votes:
        counts[target] = counts.get(target, 0) + 1
    rows = [
        f"<div><strong>{html.escape(target)}</strong>: {count}</div>"
        for target, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    details = "".join(
        f"<div>{html.escape(voter)} -> {html.escape(target)}</div>"
        for voter, target in votes
    )
    return (
        "<div class='meeting-vote-summary'>"
        "<div class='meeting-vote-title'>Voting</div>"
        "<div class='meeting-vote-counts'>"
        + "".join(rows)
        + "</div>"
        "<details><summary>Vote details</summary>"
        + details
        + "</details>"
        "</div>"
    )


def render_lobby_preview() -> str:
    return """
<div class="game-shell lobby-preview">
  <div class="topbar">
    <div><strong>Lobby</strong> | configure players</div>
    <div class="progress-wrap">
      <span>Task progress</span>
      <div class="progress-track"><div class="progress-fill" style="width:0%"></div></div>
      <strong>0.0%</strong>
    </div>
  </div>
  <div class="meeting-banner preview">
    <div class="meeting-icon">!</div>
    <div>
      <div class="meeting-title">Emergency meeting style preview</div>
      <div class="meeting-meta">This visual state appears during discussion and voting phases.</div>
    </div>
  </div>
  <div class="lobby-copy">
    <div class="lobby-title">Ready room</div>
    <div class="lobby-subtitle">Add players, choose models, and tune personality prompts before configuring the map.</div>
  </div>
</div>
"""


def render_log(snapshot: GradioSnapshot | None) -> str:
    if snapshot is None:
        return "<div class='side-log'><h3>Thoughts and actions</h3><p>No game is running.</p></div>"
    grouped: dict[int, list[dict[str, Any]]] = {}
    for decision in snapshot.decisions:
        grouped.setdefault(int(decision.get("timestep") or 0), []).append(decision)
    lines = ["<div class='side-log'><h3>Thoughts and actions</h3>"]
    for timestep in sorted(grouped):
        lines.append(f"<details open><summary>Step {timestep}</summary>")
        for decision in grouped[timestep]:
            raw_player = str(decision.get("player", ""))
            player = html.escape(player_display_name(raw_player, snapshot.players))
            color = html.escape(player_color(raw_player, snapshot.players))
            thought = html.escape(str(decision.get("thought", "")))
            action = html.escape(str(decision.get("chosen_action", "")))
            role = html.escape(player_role_label(raw_player, snapshot.players))
            action_kind = html.escape(action.split(" ", 1)[0] if action else "ACTION")
            lines.append(
                "<div class='decision-card'>"
                f"<div class='decision-head'><span class='decision-player' style='color:{color}'>"
                f"<span class='player-dot' style='background:{color}'></span>{player}</span>"
                f"{role_pill_html(role)}</div>"
                "<div class='thought-block'><div class='block-label'>Thought</div>"
                f"<div class='thought-text'>{thought}</div></div>"
                "<div class='action-block'>"
                f"<span class='action-kind'>{action_kind}</span>"
                f"<span class='action-text'>{action}</span></div>"
                "</div>"
            )
        lines.append("</details>")
    if snapshot.report_text:
        lines.append(f"<div class='winner'>{html.escape(snapshot.report_text)}</div>")
    lines.append("</div>")
    return "".join(lines)


def render_replay_log(game_log: dict[str, Any], event_index: int) -> str:
    if not game_log:
        return render_log(None)
    events = game_log.get("events", [])
    visible_events = events[: event_index + 1]
    players_by_id = {player["id"]: player for player in game_log.get("players", [])}
    lines = ["<div class='side-log'><h3>Replay</h3>"]
    for event in visible_events:
        timestep = html.escape(str(event.get("timestep", "")))
        if event.get("type") == "agent_decision":
            player_id = html.escape(str(event.get("player_id", "")))
            player_meta = players_by_id.get(str(event.get("player_id", "")), {})
            label = html.escape(
                f"{player_meta.get('name', player_id).split(':', 1)[0]}: "
                f"{short_model_name(player_meta_model(player_meta))}"
            )
            color = html.escape(str(player_meta.get("color", "#f59f00")))
            role = html.escape(str(player_meta.get("role", "unknown")))
            thought = html.escape(str(event.get("thought", "")))
            action = html.escape(str(event.get("chosen_action", "")))
            action_kind = html.escape(action.split(" ", 1)[0] if action else "ACTION")
            lines.append(
                "<div class='decision-card'>"
                f"<div class='decision-head'><span class='decision-player' style='color:{color}'>"
                f"<span class='player-dot' style='background:{color}'></span>{label}</span>"
                f"{role_pill_html(role)}</div>"
                "<div class='thought-block'><div class='block-label'>Thought</div>"
                f"<div class='thought-text'>{thought}</div></div>"
                "<div class='action-block'>"
                f"<span class='action-kind'>{action_kind}</span>"
                f"<span class='action-text'>{action}</span></div>"
                "</div>"
            )
        elif event.get("type") == "action":
            public_text = str(event.get("public_text", ""))
            for player in players_by_id.values():
                public_text = public_text.replace(
                    str(player.get("name", "")),
                    f"{str(player.get('name', '')).split(':', 1)[0]}: {short_model_name(player_meta_model(player))}",
                )
            text = html.escape(public_text)
            lines.append(f"<div class='activity'>Step {timestep}: {text}</div>")
    lines.append("</div>")
    return "".join(lines)


def replay_state_event_indices(game_log: dict[str, Any]) -> list[int]:
    if not game_log:
        return []
    return [
        index
        for index, event in enumerate(game_log.get("events", []))
        if event.get("type") == "state"
    ]


def replay_step_to_event_index(game_log: dict[str, Any], replay_step: int) -> int:
    state_indices = replay_state_event_indices(game_log)
    if not state_indices:
        return 0
    step = max(0, min(int(replay_step), len(state_indices) - 1))
    return state_indices[step]


def winner_from_game_log(game_log: dict[str, Any] | None) -> str:
    if not game_log:
        return ""
    return str(game_log.get("game", {}).get("winner") or "")


def snapshot_from_state_event(game_log: dict[str, Any], event_index: int) -> GradioSnapshot | None:
    if not game_log:
        return None
    events = game_log.get("events", [])
    state_event = None
    visible_events = events[: event_index + 1]
    for event in visible_events:
        if event.get("type") == "state":
            state_event = event
    if not state_event:
        return None
    players_by_id = {player["id"]: player for player in game_log.get("players", [])}
    players = []
    for state in state_event.get("players", []):
        player = players_by_id.get(state["id"], {})
        match = re.search(r"(\d+)$", str(state.get("id", "")))
        players.append(
            {
                "display_id": match.group(1) if match else len(players) + 1,
                "name": player.get("name", state["id"]),
                "role": player.get("role", ""),
                "alive": state.get("alive", True),
                "reported_death": state.get("reported_death", False),
                "location": state.get("location", "Cafeteria"),
                "color": player.get("color", "gray"),
                "model": player.get("model") or player.get("config", {}).get("model_id", ""),
            }
        )
    decisions = []
    activities = []
    for event in visible_events:
        player_meta = players_by_id.get(str(event.get("player_id", "")), {})
        player_name = player_meta.get("name", str(event.get("player_id", "")))
        if event.get("type") == "agent_decision":
            decisions.append(
                {
                    "player": player_name,
                    "timestep": event.get("timestep"),
                    "phase": event.get("phase"),
                    "memory": event.get("memory", ""),
                    "thought": event.get("thought", ""),
                    "requested_action": event.get("requested_action", ""),
                    "chosen_action": event.get("chosen_action", ""),
                    "speech_message": event.get("speech_message", ""),
                }
            )
        elif event.get("type") == "action":
            action_payload = event.get("action", {})
            if action_payload.get("kind") == "SPEAK":
                action_text = "SPEAK: " + str(action_payload.get("message", ""))
            else:
                action_text = str(action_payload.get("text", ""))
            activities.append(
                {
                    "timestep": event.get("timestep"),
                    "phase": event.get("phase"),
                    "round": event.get("round"),
                    "player": player_name,
                    "action": action_text,
                }
            )

    return GradioSnapshot(
        timestep=int(state_event.get("timestep") or 0),
        phase=str(state_event.get("phase", "")),
        task_progress=float(state_event.get("task_progress") or 0),
        map_payload=game_log.get("map", {}),
        players=players,
        decisions=decisions,
        activities=activities,
        meeting_info=state_event.get("meeting_info", {}),
    )


def create_app():
    import gradio as gr

    try:
        kbench = load_kbench()
        choices = model_choices(kbench)
        load_status = f"Loaded {len(choices)} kbench models."
    except Exception as exc:
        kbench = None
        choices = list(DEFAULT_MODEL_IDS)
        load_status = f"kbench is not loaded yet: {exc}"

    initial_state = {
        "players": default_players(choices),
        "rooms": default_rooms(),
        "corridors": default_edges(DEFAULT_CONNECTIONS),
        "vents": default_edges(DEFAULT_VENT_CONNECTIONS),
        "tasks": default_tasks(),
        "settings": {
            "num_common_tasks": 1,
            "num_short_tasks": 1,
            "num_long_tasks": 1,
            "discussion_rounds": 2,
            "max_num_buttons": 1,
            "kill_cooldown": 3,
            "max_timesteps": 12,
            "seed": 7,
        },
        "game_log": None,
        "replay_index": 0,
        "winner": "",
        "run_id": None,
    }

    css = """
:root {
  --ka-bg: #f6f7f9;
  --ka-panel: #ffffff;
  --ka-panel-2: #f9fafb;
  --ka-text: #121417;
  --ka-muted: #667085;
  --ka-border: #d7dde5;
  --ka-soft-border: #e6ebf1;
  --ka-map: #f1f3f5;
  --ka-room: #ffffff;
  --ka-corridor: #aeb6c0;
  --ka-vent: #d00000;
  --ka-accent: #0ca678;
  --ka-accent-soft: #d3f9d8;
  --ka-warn: #f59f00;
  --ka-danger: #e03131;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ka-bg: #0f1115;
    --ka-panel: #17191f;
    --ka-panel-2: #111318;
    --ka-text: #f1f3f5;
    --ka-muted: #a6adbb;
    --ka-border: #303641;
    --ka-soft-border: #242a34;
    --ka-map: #11141a;
    --ka-room: #1c2028;
    --ka-corridor: #4b5563;
    --ka-vent: #ff6b6b;
    --ka-accent: #12b886;
    --ka-accent-soft: #123226;
    --ka-warn: #fab005;
    --ka-danger: #ff6b6b;
  }
}
.gradio-container {
  max-width: none !important;
  width: 100% !important;
  padding-left: 16px !important;
  padding-right: 16px !important;
}
body, .gradio-container { overflow-x: hidden; }
.game-shell, .empty-map {
  width: 100%;
  box-sizing: border-box;
  min-height: 620px;
  background: var(--ka-bg);
  color: var(--ka-text);
  border: 1px solid var(--ka-border);
  border-radius: 8px;
  padding: 12px;
}
.empty-map { display:flex; align-items:center; justify-content:center; color:var(--ka-muted); font-size:18px; }
.lobby-preview { display:flex; flex-direction:column; gap:16px; }
.lobby-copy { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; color:var(--ka-text); }
.lobby-title { font-size:32px; font-weight:800; }
.lobby-subtitle { max-width:520px; color:var(--ka-muted); margin-top:10px; }
.topbar {
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:16px;
  padding:10px 12px;
  background:var(--ka-panel);
  color:var(--ka-text);
  border:1px solid var(--ka-border);
  border-radius:8px;
  margin-bottom:10px;
}
.progress-wrap { display:grid; grid-template-columns:auto minmax(140px, 220px) auto; align-items:center; gap:10px; color:var(--ka-text); }
.progress-track { height:10px; border-radius:999px; background:var(--ka-soft-border); overflow:hidden; border:1px solid var(--ka-border); }
.progress-fill { height:100%; background:linear-gradient(90deg, var(--ka-accent), #74c0fc); border-radius:999px; }
.phase { padding:10px 12px; border-radius:8px; margin:8px 0 12px; font-weight:700; }
.phase.task { background:#dff1ff; color:#063f66; }
.phase.voting { background:#ffe3e3; color:#8f1d1d; }
@media (prefers-color-scheme: dark) {
  .phase.task { background:#10273a; color:#9bd3ff; }
  .phase.voting { background:#341313; color:#ffc9c9; }
}
.meeting-banner {
  display:flex;
  align-items:center;
  gap:14px;
  padding:14px 16px;
  border:1px solid rgba(224,49,49,.55);
  background:linear-gradient(90deg, rgba(224,49,49,.18), rgba(245,159,0,.08));
  color:var(--ka-text);
  border-radius:10px;
  margin:8px 0 12px;
}
.meeting-banner.preview { margin-top:4px; }
.meeting-icon {
  width:38px;
  height:38px;
  border-radius:8px;
  display:flex;
  align-items:center;
  justify-content:center;
  background:rgba(224,49,49,.16);
  color:var(--ka-danger);
  border:1px solid rgba(224,49,49,.55);
  font-weight:900;
}
.meeting-title { text-transform:uppercase; letter-spacing:.04em; font-weight:850; font-size:18px; }
.meeting-meta { color:var(--ka-muted); font-size:13px; margin-top:2px; }
.outcome-banner {
  display:flex;
  align-items:center;
  gap:14px;
  padding:14px 16px;
  border-radius:10px;
  margin:10px 0 12px;
  background:#111827 !important;
  color:#ffffff !important;
  border:1px solid #4b5563 !important;
  box-shadow:0 8px 24px rgba(0,0,0,.18);
}
.outcome-banner.crew { border-color:rgba(18,184,134,.7) !important; background:linear-gradient(90deg, #052e22, #111827) !important; }
.outcome-banner.impostor { border-color:rgba(255,107,107,.7) !important; background:linear-gradient(90deg, #3b1111, #111827) !important; }
.outcome-icon {
  width:42px;
  height:42px;
  border-radius:10px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight:950;
  background:rgba(255,255,255,.09);
  border:1px solid rgba(255,255,255,.28);
  flex:0 0 auto;
}
.outcome-title { text-transform:uppercase; letter-spacing:.04em; font-weight:950; font-size:18px; }
.outcome-message { color:#e5e7eb; margin-top:2px; }
.map-svg { width:100%; height:auto; min-height:540px; background:var(--ka-map); border-radius:8px; overflow:visible; }
.corridor { stroke:var(--ka-corridor); stroke-width:6; stroke-linecap:round; }
.vent { stroke:var(--ka-vent); stroke-width:3; stroke-dasharray:8 7; stroke-linecap:round; }
.room { fill:var(--ka-room); stroke:var(--ka-text); stroke-width:1.5; }
.room-label { text-anchor:middle; font-size:13px; font-weight:750; fill:var(--ka-text); }
.task-label { text-anchor:middle; font-size:11px; fill:var(--ka-muted); }
.player { stroke:var(--ka-text); stroke-width:1.6; }
.dead { stroke:var(--ka-danger); stroke-width:3; opacity:.85; }
.dead-x { stroke:#ffffff; stroke-width:2.3; stroke-linecap:round; filter:drop-shadow(0 0 2px #000); }
.meeting-stage {
  display:grid;
  grid-template-columns:minmax(420px, 1.2fr) minmax(340px, .8fr);
  gap:12px;
  min-height:540px;
}
.meeting-map-pane {
  min-height:540px;
  background:var(--ka-map);
  border-radius:8px;
  overflow:hidden;
}
.meeting-stage .map-svg { min-height:540px; }
@media (max-width: 1100px) {
  .meeting-stage { grid-template-columns:1fr; }
  .meeting-conversation { max-height:380px; }
}
.side-log {
  max-height:735px;
  overflow:auto;
  padding:12px;
  background:var(--ka-panel);
  color:var(--ka-text);
  border:1px solid var(--ka-border);
  border-radius:8px;
}
.side-log h3 { margin-top:0; color:var(--ka-text); }
.decision-card { border:1px solid var(--ka-border); border-radius:8px; padding:10px; margin:10px 0; background:var(--ka-panel-2); }
.decision-head { display:flex; justify-content:space-between; gap:10px; align-items:center; margin-bottom:8px; }
.decision-player { font-weight:800; color:var(--ka-warn); display:inline-flex; align-items:center; gap:8px; }
.player-dot { width:10px; height:10px; border-radius:999px; display:inline-block; border:1px solid var(--ka-text); flex:0 0 auto; }
.step-pill { font-size:11px; color:var(--ka-muted); border:1px solid var(--ka-border); border-radius:5px; padding:3px 7px; text-transform:uppercase; }
.role-pill { font-weight:900; letter-spacing:.04em; }
.role-impostor { color:#ff6b6b; border-color:rgba(255,107,107,.65); background:rgba(224,49,49,.13); }
.role-crewmate { color:#63e6be; border-color:rgba(99,230,190,.65); background:rgba(18,184,134,.13); }
.thought-block { border-left:3px solid var(--ka-warn); padding-left:10px; margin:8px 0 10px; }
.block-label { text-transform:uppercase; letter-spacing:.08em; font-size:11px; color:var(--ka-warn); font-weight:800; margin-bottom:6px; }
.thought-text { color:var(--ka-muted); font-style:italic; line-height:1.55; background:var(--ka-panel); border:1px solid var(--ka-soft-border); border-radius:6px; padding:10px; }
.action-block { display:flex; align-items:center; gap:10px; background:var(--ka-panel); border:1px solid var(--ka-soft-border); border-radius:6px; padding:8px 10px; }
.action-kind { color:#8ce99a; background:#063b2b; border-radius:5px; font-size:11px; font-weight:800; padding:4px 8px; text-transform:uppercase; }
.action-text { color:var(--ka-accent); font-weight:750; }
.activity { border-bottom:1px solid var(--ka-soft-border); padding:7px 0; color:var(--ka-muted); }
.meeting-conversation { max-height:540px; overflow-y:auto; overflow-x:hidden; background:var(--ka-panel); color:var(--ka-text); border:1px solid var(--ka-border); border-radius:10px; padding:12px; }
.meeting-conversation.full {
  min-height:540px;
  max-height:720px;
  width:100%;
  box-sizing:border-box;
}
.meeting-conversation-title { text-transform:uppercase; letter-spacing:.06em; color:var(--ka-muted); font-size:12px; font-weight:850; margin-bottom:8px; }
.meeting-bubble { display:grid; grid-template-columns:14px 1fr; gap:10px; padding:10px; margin:8px 0; background:var(--ka-panel-2); border:1px solid var(--ka-soft-border); border-radius:8px; }
.vote-bubble { border-color:rgba(250,176,5,.45); background:rgba(245,159,0,.08); }
.meeting-speaker-dot { width:12px; height:12px; border-radius:999px; margin-top:4px; border:1px solid var(--ka-text); }
.meeting-speaker { font-weight:850; margin-bottom:4px; }
.meeting-message { color:var(--ka-text); line-height:1.55; }
.vote-chip { display:inline-block; margin-right:8px; padding:2px 7px; border-radius:5px; background:rgba(245,159,0,.18); color:var(--ka-warn); border:1px solid rgba(245,159,0,.45); font-size:11px; font-weight:900; }
.meeting-vote-summary { margin:8px 0 12px; padding:10px; border-radius:8px; border:1px solid rgba(245,159,0,.45); background:rgba(245,159,0,.08); }
.meeting-vote-title { text-transform:uppercase; letter-spacing:.06em; color:var(--ka-warn); font-size:12px; font-weight:900; margin-bottom:6px; }
.meeting-vote-counts { display:flex; flex-wrap:wrap; gap:10px 18px; margin-bottom:6px; }
.meeting-vote-summary details { color:var(--ka-muted); }
.winner {
  margin-top:10px;
  padding:10px;
  border-radius:6px;
  background:#111827 !important;
  color:#ffffff !important;
  font-weight:800;
  border:1px solid #4b5563 !important;
}
"""

    with gr.Blocks(title="kbench_amongus") as demo:
        gr.HTML(f"<style>{css}</style>")
        state = gr.State(initial_state)
        latest_snapshot = gr.State(None)

        with gr.Row():
            with gr.Column(scale=3):
                map_html = gr.HTML(render_lobby_preview())
            with gr.Column(scale=2):
                side_html = gr.HTML(render_log(None))

        with gr.Group(visible=False) as lobby_scene:
            gr.HTML("")

        with gr.Group(visible=True) as config_scene:
            gr.Markdown("## Game Config")
            with gr.Row():
                with gr.Column(scale=7):
                    with gr.Tabs():
                        with gr.Tab("Lobby"):
                            player_count = gr.Slider(2, MAX_PLAYERS, value=5, step=1, label="Number of players")
                            randomize_all_names = gr.Button("Randomize All Names")
                            player_components = []
                            for index in range(MAX_PLAYERS):
                                initial_player = (
                                    initial_state["players"][index]
                                    if index < len(initial_state["players"])
                                    else {
                                        "name": random_player_alias(index),
                                        "role": "crewmate",
                                        "model": choices[min(index, len(choices) - 1)] if choices else "",
                                    }
                                )
                                with gr.Row(visible=index < 5) as row:
                                    enabled = gr.Checkbox(value=index < 5, label=f"P{index + 1}")
                                    name = gr.Textbox(value=initial_player["name"], label="Name")
                                    random_name = gr.Button("Random")
                                    role = gr.Dropdown(["crewmate", "impostor"], value=initial_player["role"], label="Role")
                                    model = gr.Dropdown(choices, value=initial_player["model"], label="Model")
                                personality = gr.Textbox(
                                    value="",
                                    label=f"Player {index + 1} personality",
                                    placeholder="Add extra instruction or personality.",
                                    lines=2,
                                    visible=index < 5,
                                )
                                player_components.append((row, enabled, name, role, model, personality, random_name))

                        with gr.Tab("Rules"):
                            with gr.Row():
                                num_common = gr.Number(value=1, precision=0, label="Common task types selected")
                                num_short = gr.Number(value=1, precision=0, label="Short tasks per crewmate")
                                num_long = gr.Number(value=1, precision=0, label="Long tasks per crewmate")
                                discussion_rounds = gr.Number(value=2, precision=0, label="Discussion rounds")
                            with gr.Row():
                                max_buttons = gr.Number(value=1, precision=0, label="Max emergency buttons")
                                kill_cooldown = gr.Number(value=3, precision=0, label="Kill cooldown")
                                max_timesteps = gr.Number(value=12, precision=0, label="Max timesteps")
                                seed = gr.Number(value=7, precision=0, label="Seed")

                        with gr.Tab("Rooms"):
                            room_count = gr.Slider(2, MAX_ROOMS, value=min(len(DEFAULT_ROOM_DATA), MAX_ROOMS), step=1, label="Enabled room rows")
                            room_components = []
                            default_room_names = list(DEFAULT_ROOM_DATA)[:MAX_ROOMS]
                            for index in range(MAX_ROOMS):
                                room_name = default_room_names[index] if index < len(default_room_names) else ""
                                with gr.Row(visible=index < len(default_room_names)) as row:
                                    enabled = gr.Checkbox(value=True, label=f"Room {index + 1}")
                                    name = gr.Textbox(value=room_name, label="Room name")
                                room_components.append((row, enabled, name))

                        with gr.Tab("Corridors"):
                            corridor_count = gr.Slider(0, MAX_EDGES, value=min(len(DEFAULT_CONNECTIONS), MAX_EDGES), step=1, label="Enabled corridor rows")
                            corridor_components = []
                            for index in range(MAX_EDGES):
                                default = DEFAULT_CONNECTIONS[index] if index < len(DEFAULT_CONNECTIONS) else ("", "")
                                with gr.Row(visible=index < len(DEFAULT_CONNECTIONS)) as row:
                                    enabled = gr.Checkbox(value=index < len(DEFAULT_CONNECTIONS), label=f"Corridor {index + 1}")
                                    room_a = gr.Textbox(value=default[0], label="From")
                                    room_b = gr.Textbox(value=default[1], label="To")
                                corridor_components.append((row, enabled, room_a, room_b))

                        with gr.Tab("Vents"):
                            vent_count = gr.Slider(0, MAX_EDGES, value=min(len(DEFAULT_VENT_CONNECTIONS), MAX_EDGES), step=1, label="Enabled vent rows")
                            vent_components = []
                            for index in range(MAX_EDGES):
                                default = DEFAULT_VENT_CONNECTIONS[index] if index < len(DEFAULT_VENT_CONNECTIONS) else ("", "")
                                with gr.Row(visible=index < len(DEFAULT_VENT_CONNECTIONS)) as row:
                                    enabled = gr.Checkbox(value=index < len(DEFAULT_VENT_CONNECTIONS), label=f"Vent {index + 1}")
                                    room_a = gr.Textbox(value=default[0], label="From")
                                    room_b = gr.Textbox(value=default[1], label="To")
                                vent_components.append((row, enabled, room_a, room_b))

                        with gr.Tab("Tasks"):
                            task_defaults = default_tasks()[:MAX_TASKS]
                            task_count = gr.Slider(0, MAX_TASKS, value=len(task_defaults), step=1, label="Enabled task rows")
                            task_components = []
                            for index in range(MAX_TASKS):
                                default = task_defaults[index] if index < len(task_defaults) else {
                                    "enabled": False,
                                    "name": "",
                                    "room": "Cafeteria",
                                    "type": "short",
                                    "duration": 1,
                                }
                                with gr.Row(visible=index < len(task_defaults)) as row:
                                    enabled = gr.Checkbox(value=default["enabled"], label=f"Task {index + 1}")
                                    task_name = gr.Textbox(value=default["name"], label="Task name")
                                    task_room = gr.Textbox(value=default["room"], label="Room")
                                    task_type = gr.Dropdown(TASK_TYPES, value=default["type"], label="Type")
                                    duration = gr.Number(value=default["duration"], precision=0, label="Duration")
                                task_components.append((row, enabled, task_name, task_room, task_type, duration))

                with gr.Column(scale=3):
                    validation = gr.Markdown("")
                    with gr.Row():
                        validate_btn = gr.Button("Validate Config")
                        export_btn = gr.Button("Export Config")
                    play_btn = gr.Button("Play", variant="primary")
                    export_box = gr.Code(
                        label="Copyable Python config",
                        language="python",
                        lines=22,
                        buttons=["copy", "download"],
                        value="# Click Export Config to generate a copyable snippet.",
                    )

        with gr.Group(visible=False) as gameplay_scene:
            gr.Markdown("## Gameplay")
            with gr.Row():
                prev_btn = gr.Button("Prev")
                next_btn = gr.Button("Next")
                replay_pos = gr.Number(value=0, precision=0, label="Replay step")
                stop_btn = gr.Button("Stop", variant="stop")
                restart_btn = gr.Button("Back to Config")

        def update_player_visibility(count):
            count = int(count)
            updates = []
            for index in range(MAX_PLAYERS):
                visible = index < count
                updates.extend([
                    gr.update(visible=visible),
                    gr.update(value=visible),
                    gr.update(visible=visible),
                ])
            return updates

        player_count.change(
            update_player_visibility,
            inputs=player_count,
            outputs=[
                output
                for row, enabled, _name, _role, _model, personality, _random_name in player_components
                for output in (row, enabled, personality)
            ],
        )

        for index, (_row, _enabled, name, _role, _model, _personality, random_name) in enumerate(player_components):
            random_name.click(
                lambda index=index: random_player_alias(index),
                outputs=name,
            )

        randomize_all_names.click(
            lambda: [random_player_alias(index) for index in range(MAX_PLAYERS)],
            outputs=[name for _row, _enabled, name, _role, _model, _personality, _random_name in player_components],
        )

        def update_row_visibility(count, total, include_enabled=True):
            count = int(count)
            updates = []
            for index in range(total):
                visible = index < count
                updates.append(gr.update(visible=visible))
                if include_enabled:
                    updates.append(gr.update(value=visible))
            return updates

        room_count.change(
            lambda count: update_row_visibility(count, MAX_ROOMS),
            inputs=room_count,
            outputs=[output for row, enabled, _name in room_components for output in (row, enabled)],
        )
        corridor_count.change(
            lambda count: update_row_visibility(count, MAX_EDGES),
            inputs=corridor_count,
            outputs=[output for row, enabled, _a, _b in corridor_components for output in (row, enabled)],
        )
        vent_count.change(
            lambda count: update_row_visibility(count, MAX_EDGES),
            inputs=vent_count,
            outputs=[output for row, enabled, _a, _b in vent_components for output in (row, enabled)],
        )
        task_count.change(
            lambda count: update_row_visibility(count, MAX_TASKS),
            inputs=task_count,
            outputs=[output for row, enabled, *_rest in task_components for output in (row, enabled)],
        )

        def collect_state(*values):
            cursor = 0
            players = []
            for _ in range(MAX_PLAYERS):
                enabled, name, role, model, personality = values[cursor: cursor + 5]
                cursor += 5
                players.append(
                    {
                        "enabled": bool(enabled),
                        "name": str(name).strip(),
                        "role": role,
                        "model": model,
                        "personality": str(personality or "").strip(),
                    }
                )
            settings = {
                "num_common_tasks": int(values[cursor]),
                "num_short_tasks": int(values[cursor + 1]),
                "num_long_tasks": int(values[cursor + 2]),
                "discussion_rounds": int(values[cursor + 3]),
                "max_num_buttons": int(values[cursor + 4]),
                "kill_cooldown": int(values[cursor + 5]),
                "max_timesteps": int(values[cursor + 6]),
                "seed": int(values[cursor + 7]),
            }
            cursor += 8
            rooms = []
            for _ in range(MAX_ROOMS):
                enabled, name = values[cursor: cursor + 2]
                cursor += 2
                rooms.append({"enabled": bool(enabled), "name": str(name).strip()})
            corridors = []
            for _ in range(MAX_EDGES):
                enabled, room_a, room_b = values[cursor: cursor + 3]
                cursor += 3
                corridors.append(
                    {"enabled": bool(enabled), "from": str(room_a).strip(), "to": str(room_b).strip()}
                )
            vents = []
            for _ in range(MAX_EDGES):
                enabled, room_a, room_b = values[cursor: cursor + 3]
                cursor += 3
                vents.append(
                    {"enabled": bool(enabled), "from": str(room_a).strip(), "to": str(room_b).strip()}
                )
            tasks = []
            for _ in range(MAX_TASKS):
                enabled, name, room, task_type, duration = values[cursor: cursor + 5]
                cursor += 5
                tasks.append(
                    {
                        "enabled": bool(enabled),
                        "name": str(name).strip(),
                        "room": str(room).strip(),
                        "type": task_type,
                        "duration": int(duration),
                    }
                )
            return {
                "players": players,
                "settings": settings,
                "rooms": rooms,
                "corridors": corridors,
                "vents": vents,
                "tasks": tasks,
                "game_log": None,
                "replay_index": 0,
                "winner": "",
                "run_id": None,
            }

        collect_inputs = (
            [component for _row, enabled, name, role, model, personality, _random_name in player_components for component in (enabled, name, role, model, personality)]
            + [num_common, num_short, num_long, discussion_rounds, max_buttons, kill_cooldown, max_timesteps, seed]
            + [component for _row, enabled, name in room_components for component in (enabled, name)]
            + [component for _row, enabled, room_a, room_b in corridor_components for component in (enabled, room_a, room_b)]
            + [component for _row, enabled, room_a, room_b in vent_components for component in (enabled, room_a, room_b)]
            + [component for _row, enabled, task_name, task_room, task_type, duration in task_components for component in (enabled, task_name, task_room, task_type, duration)]
        )

        def validate_only(*values):
            new_state = collect_state(*values)
            try:
                if kbench is None:
                    raise ValueError("kbench could not be loaded. Check kaggle-benchmarks/.env.")
                make_game_config(kbench, new_state)
            except Exception as exc:
                return new_state, f"### Validation failed\n```text\n{exc}\n```"
            return new_state, "### Validation passed"

        validate_btn.click(
            validate_only,
            inputs=collect_inputs,
            outputs=[state, validation],
        )

        def export_only(*values):
            new_state = collect_state(*values)
            try:
                code = export_config_code(new_state)
            except Exception as exc:
                return new_state, f"### Export failed\n```text\n{exc}\n```", ""
            return new_state, "### Export ready", code

        export_btn.click(
            export_only,
            inputs=collect_inputs,
            outputs=[state, validation, export_box],
        )

        def run_game_stream(*values):
            if kbench is None:
                yield (
                    collect_state(*values),
                    None,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    empty_map_html("kbench is not loaded."),
                    "<div class='side-log'><h3>Error</h3><p>Check kaggle-benchmarks/.env.</p></div>",
                    0,
                )
                return
            new_state = collect_state(*values)
            run_id = str(time.time_ns())
            stop_event = threading.Event()
            RUN_STOP_EVENTS[run_id] = stop_event
            new_state["run_id"] = run_id
            try:
                game_config = make_game_config(kbench, new_state)
            except Exception as exc:
                RUN_STOP_EVENTS.pop(run_id, None)
                yield (
                    new_state,
                    None,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    empty_map_html("Validation failed."),
                    f"<div class='side-log'><h3>Validation failed</h3><pre>{html.escape(str(exc))}</pre></div>",
                    0,
                )
                return

            updates: queue.Queue[Any] = queue.Queue()
            ui = GradioGameUI(updates, stop_event=stop_event)
            result_box: dict[str, Any] = {}

            def target():
                try:
                    game = ConfiguredAmongUs(game_config=game_config, UI=ui)
                    winner = game.run_game()
                    result_box["result"] = game.result_summary(winner)
                except GameStopped:
                    result_box["stopped"] = True
                except Exception as exc:
                    result_box["error"] = exc
                finally:
                    updates.put(None)

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            last_snapshot = None
            while True:
                item = updates.get()
                if item is None:
                    break
                last_snapshot = item
                if stop_event.is_set():
                    result_box["stopped"] = True
                    break
                yield (
                    new_state,
                    last_snapshot,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    render_map(last_snapshot),
                    render_log(last_snapshot),
                    0,
                )
            RUN_STOP_EVENTS.pop(run_id, None)
            if result_box.get("stopped"):
                if last_snapshot is not None:
                    last_snapshot.report_text = "Game stopped by user."
                yield (
                    new_state,
                    last_snapshot,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    render_map(last_snapshot),
                    "<div class='side-log'><h3>Stopped</h3><p>Game stopped by user. In-flight LLM calls may finish in the background.</p></div>",
                    0,
                )
                return
            if "error" in result_box:
                error = result_box["error"]
                yield (
                    new_state,
                    last_snapshot,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    render_map(last_snapshot),
                    f"<div class='side-log'><h3>Game failed</h3><pre>{html.escape(str(error))}</pre></div>",
                    0,
                )
                return
            result = result_box.get("result", {})
            new_state["game_log"] = result.get("game_log")
            new_state["winner"] = result.get("winner", "")
            replay_steps = replay_state_event_indices(new_state["game_log"])
            new_state["replay_index"] = max(0, len(replay_steps) - 1)
            event_index = replay_step_to_event_index(new_state["game_log"], new_state["replay_index"])
            final_snapshot = snapshot_from_state_event(new_state["game_log"], event_index) or last_snapshot
            if final_snapshot and not final_snapshot.report_text:
                final_snapshot.report_text = new_state["winner"]
            yield (
                new_state,
                final_snapshot,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                render_map(final_snapshot, new_state["replay_index"]),
                render_replay_log(new_state["game_log"], event_index),
                new_state["replay_index"],
            )

        play_btn.click(
            run_game_stream,
            inputs=collect_inputs,
            outputs=[
                state,
                latest_snapshot,
                lobby_scene,
                config_scene,
                gameplay_scene,
                map_html,
                side_html,
                replay_pos,
            ],
        )

        def stop_game(state_value, snapshot_value):
            state_value = state_value or {}
            run_id = state_value.get("run_id")
            state_value["stop_requested"] = True
            if run_id and run_id in RUN_STOP_EVENTS:
                RUN_STOP_EVENTS[run_id].set()
            if snapshot_value is not None:
                snapshot_value.report_text = "Stop requested. Waiting for the current LLM/action to finish."
            return (
                state_value,
                snapshot_value,
                render_map(snapshot_value),
                "<div class='side-log'><h3>Stop requested</h3><p>Waiting for the current LLM/action boundary. New game updates are ignored after the stop signal.</p></div>",
                0,
            )

        stop_btn.click(
            stop_game,
            inputs=[state, latest_snapshot],
            outputs=[state, latest_snapshot, map_html, side_html, replay_pos],
        )

        def replay_move(state_value, delta):
            game_log = (state_value or {}).get("game_log")
            if not game_log:
                return state_value, None, empty_map_html("No replay is available."), render_log(None), 0
            state_indices = replay_state_event_indices(game_log)
            if not state_indices:
                return state_value, None, empty_map_html("No replay states are available."), render_log(None), 0
            step = int((state_value or {}).get("replay_index", 0)) + delta
            step = max(0, min(step, len(state_indices) - 1))
            event_index = state_indices[step]
            state_value["replay_index"] = step
            snap = snapshot_from_state_event(game_log, event_index)
            if snap is not None:
                snap.report_text = str(state_value.get("winner") or winner_from_game_log(game_log))
            return (
                state_value,
                snap,
                render_map(snap, step),
                render_replay_log(game_log, event_index),
                step,
            )

        prev_btn.click(
            lambda state_value: replay_move(state_value, -1),
            inputs=state,
            outputs=[state, latest_snapshot, map_html, side_html, replay_pos],
        )
        next_btn.click(
            lambda state_value: replay_move(state_value, 1),
            inputs=state,
            outputs=[state, latest_snapshot, map_html, side_html, replay_pos],
        )
        restart_btn.click(
            lambda: (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                empty_map_html("Game configuration"),
                "<div class='side-log'><h3>Config summary</h3><p>Edit settings and press Play again.</p></div>",
            ),
            outputs=[lobby_scene, config_scene, gameplay_scene, map_html, side_html],
        )

    return demo


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}.")


def main():
    parser = argparse.ArgumentParser(description="Launch the kbench_amongus Gradio app.")
    parser.add_argument(
        "--share",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Create a public Gradio share link. Default: False.",
    )
    parser.add_argument("--server-name", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args()
    app = create_app()
    app.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
