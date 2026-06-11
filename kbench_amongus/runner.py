from __future__ import annotations

import random
import re
from typing import Any

import numpy as np

from kbench_amongus.agent.base import AgentContext
from kbench_amongus.agent.llm_default import DefaultLLMAgent
from kbench_amongus.agent.validation import (
    InvalidAgentError,
    bind_and_validate_agent,
    is_raw_kbench_llm,
)
from kbench_amongus.config import GameConfig
from kbench_amongus.envs.action import (
    CallMeeting,
    CompleteFakeTask,
    CompleteTask,
    Kill,
    MoveTo,
    Speak,
    Vent,
    ViewMonitor,
    Vote,
)
from kbench_amongus.envs.configs.map_config import (
    map_coords,
    room_data,
)
from kbench_amongus.envs.game import AmongUs
from kbench_amongus.envs.map import Map
from kbench_amongus.envs.player import Crewmate, Impostor, PLAYER_COLORS
from kbench_amongus.envs.task import TaskAssignment


EXTRA_PLAYER_COLORS = [
    "gray",
    "olive",
    "teal",
    "navy",
    "maroon",
    "gold",
    "violet",
    "indigo",
    "coral",
    "salmon",
]


def player_id(player_or_name) -> str:
    text = getattr(player_or_name, "name", str(player_or_name))
    match = re.search(r"Player\s+(\d+)", text)
    if match:
        return f"player_{match.group(1)}"
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def normalize_role(role):
    text = str(role or "crewmate").strip().lower()
    if text == "imposter":
        text = "impostor"
    if text not in {"crewmate", "impostor"}:
        raise ValueError(f"Unsupported role {role!r}; expected crewmate or impostor")
    return text


def coerce_game_config(game_config):
    if game_config is None:
        config = GameConfig()
    elif isinstance(game_config, GameConfig):
        config = game_config
    elif isinstance(game_config, dict):
        config = GameConfig(game_settings=game_config)
    else:
        raise TypeError(
            "game_config must be a GameConfig, dict, or None; "
            f"got {type(game_config).__name__}"
        )

    if not config.player_configs:
        raise InvalidAgentError(
            "GameConfig.player_configs must define every player with "
            "{'role': ..., 'agent': ...}."
        )
    return config


def assign_player_colors(player_configs, seed):
    palette = list(PLAYER_COLORS + EXTRA_PLAYER_COLORS)
    rng = random.Random(seed)
    rng.shuffle(palette)

    used_colors = {
        str(spec["color"])
        for spec in player_configs
        if spec.get("color") is not None
    }
    available_colors = [color for color in palette if color not in used_colors]
    assigned = []
    next_auto_index = len(palette)

    for spec in player_configs:
        if spec.get("color") is not None:
            assigned.append(str(spec["color"]))
        elif available_colors:
            color = available_colors.pop(0)
            used_colors.add(color)
            assigned.append(color)
        else:
            while True:
                color = fallback_color(next_auto_index)
                next_auto_index += 1
                if color not in used_colors:
                    used_colors.add(color)
                    assigned.append(color)
                    break
    return assigned


def fallback_color(index):
    colors = PLAYER_COLORS + EXTRA_PLAYER_COLORS
    if index < len(colors):
        return colors[index]
    hue = (index * 0.618033988749895) % 1.0
    return f"#{int(hue * 255):02x}{int((1 - hue) * 180):02x}99"


class ConfiguredAmongUs(AmongUs):
    """Game runner configured only by explicit player agents."""

    def __init__(
        self,
        *,
        game_config=None,
        UI=None,
    ):
        self.experiment_config = coerce_game_config(game_config)
        self.raw_player_configs = list(self.experiment_config.player_configs)
        self._validate_player_configs()
        self.seed = self.experiment_config.seed
        self.decision_log: list[dict[str, Any]] = []
        self.state_log: list[dict[str, Any]] = []
        self.agent_metadata: dict[str, dict[str, str]] = {}
        self.resolved_player_configs: list[dict[str, Any]] = []

        super().__init__(
            game_config=self.experiment_config.resolved_game_settings(),
            include_human=False,
            test=False,
            agent_config=None,
            UI=UI,
            task_config=self.experiment_config.task_config,
        )
        self.map = Map(
            room_data=self.experiment_config.map_room_data,
            connections=self.experiment_config.map_connections,
            vent_connections=self.experiment_config.map_vent_connections,
        )

    def _validate_player_configs(self) -> None:
        num_players = len(self.raw_player_configs)
        num_impostors = 0
        for index, spec in enumerate(self.raw_player_configs, start=1):
            if "role" not in spec:
                raise ValueError(f"player_configs[{index}] is missing required role.")
            if "agent" not in spec:
                raise InvalidAgentError(
                    f"player_configs[{index}] is missing required agent."
                )
            if normalize_role(spec.get("role")) == "impostor":
                num_impostors += 1
        if num_players < 2:
            raise ValueError("At least two players are required.")
        if num_impostors < 1:
            raise ValueError("At least one impostor is required.")

    def initialize_game(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        super().initialize_game()
        self.state_log = [
            self._state_snapshot(
                timestep=0,
                phase="setup",
                summary="Initial game state",
                action_index=None,
            )
        ]

    def initialize_players(self):
        self.players = []
        self.resolved_player_configs = []
        assigned_colors = assign_player_colors(self.raw_player_configs, self.seed)
        for index, (spec, color) in enumerate(
            zip(self.raw_player_configs, assigned_colors), start=1
        ):
            role = normalize_role(spec.get("role"))
            player_name = spec.get("name") or f"Player {index}"
            player_cls = Impostor if role == "impostor" else Crewmate
            player = player_cls(
                name=player_name,
                color=color,
                location=spec.get("location", "Cafeteria"),
            )
            self.players.append(player)
            self.resolved_player_configs.append(
                {
                    "player_id": index,
                    "name": player_name,
                    "color": color,
                    "role": role,
                    "location": spec.get("location", "Cafeteria"),
                    "model_id": spec.get("model_id", ""),
                    "agent": spec["agent"],
                }
            )
        self.camera_record = {
            player.name: "stand quietly and do nothing" for player in self.players
        }
        self.task_assignment = TaskAssignment(
            self.map.ship_map,
            self.game_config,
            self.experiment_config.task_config,
        )
        self.task_assignment.assign_tasks_to_players(self.players)
        self.update_map()

    def initialize_agents(self):
        self.agents = []
        self.agent_metadata = {}
        for player, spec in zip(self.players, self.resolved_player_configs):
            agent = self._normalize_agent(spec["agent"])
            agent = bind_and_validate_agent(agent, player, self)
            self.agents.append(agent)
            self.agent_metadata[player.name] = {
                "agent_type": self._agent_type_name(agent),
                "model": self._model_name(agent),
            }

    def _normalize_agent(self, agent):
        if is_raw_kbench_llm(agent):
            return DefaultLLMAgent(agent)
        return agent

    def build_agent_context(self, player) -> AgentContext:
        visible_players = [
            {
                "name": other.name,
                "location": other.location,
                "alive": other.is_alive,
                "reported_death": other.reported_death,
            }
            for other in self.map.get_players_in_room(
                player.location, include_new_deaths=True
            )
        ]
        meeting = None
        if self.current_phase == "meeting":
            max_rounds = self.game_config["discussion_rounds"]
            current_round = max_rounds - self.discussion_rounds_left
            meeting = {
                "round": current_round,
                "rounds_left": self.discussion_rounds_left,
                "max_rounds": max_rounds,
                "votes_so_far": dict(self.vote_info_one_round),
                "meeting_info": dict(self.last_meeting_info or {}),
            }
        return AgentContext(
            player_name=player.name,
            role=player.identity,
            phase=self.current_phase,
            timestep=self.timestep,
            location=player.location,
            alive=player.is_alive,
            task_progress=self.task_assignment.check_task_completion(),
            game_settings=dict(self.game_config),
            available_actions=list(player.get_available_actions()),
            observations=list(player.observation_history[-8:]),
            action_history=[self._safe_action_record(record) for record in player.action_history[-8:]],
            assigned_tasks=[
                {
                    "name": task.name,
                    "type": task.task_type,
                    "location": task.location,
                    "duration": task.max_duration,
                    "completed": task.check_completion(),
                    "path": list(task.find_path(player.location, player.identity)),
                }
                for task in player.tasks
            ],
            visible_players=visible_players,
            meeting=meeting,
            map=self._map_payload(),
        )

    def agent_step(self, agent):
        previous_activity_count = len(self.activity_log)
        super().agent_step(agent)
        self._stamp_agent_decision(agent)
        if len(self.activity_log) > previous_activity_count:
            record = self.activity_log[-1]
            self.state_log.append(
                self._state_snapshot(
                    timestep=record["timestep"],
                    phase=record["phase"],
                    summary=f"State after action {len(self.activity_log)}",
                    action_index=len(self.activity_log) - 1,
                )
            )

    def _stamp_agent_decision(self, agent):
        log = getattr(agent, "decision_log", None)
        if not log:
            return
        latest = log[-1]
        if (
            latest.get("player") == agent.player.name
            and latest.get("timestep") is None
        ):
            latest["timestep"] = self.timestep
            latest["phase"] = self.current_phase
            self.decision_log.append(dict(latest))

    def result_summary(self, winner):
        game_log = self.build_game_log(winner)
        return {
            "winner_code": winner,
            "winner": self._winner_text(winner),
            "timesteps": self.timestep,
            "task_progress": self.task_assignment.check_task_completion(),
            "players": [
                {
                    "name": player.name,
                    "identity": player.identity,
                    "alive": player.is_alive,
                    "agent_type": self.agent_metadata.get(player.name, {}).get(
                        "agent_type", "unknown"
                    ),
                    "model": self.agent_metadata.get(player.name, {}).get("model", ""),
                }
                for player in self.players
            ],
            "activity_log": [
                {
                    **{
                        key: str(value)
                        for key, value in record.items()
                        if key in {"phase", "action", "player", "round"}
                    },
                    "timestep": record["timestep"],
                }
                for record in self.activity_log
            ],
            "important_activity_log": [
                {
                    **{
                        key: str(value)
                        for key, value in record.items()
                        if key in {"phase", "action", "player", "round"}
                    },
                    "timestep": record["timestep"],
                }
                for record in self.important_activity_log
            ],
            "decision_log": list(self.decision_log),
            "game_log": game_log,
        }

    def build_game_log(self, winner):
        events = [self._state_event_from_snapshot(1, self.state_log[0])]
        event_id = 2
        decisions_by_key = {}
        for decision in self.decision_log:
            key = (decision.get("timestep"), decision.get("player"))
            decisions_by_key.setdefault(key, []).append(decision)

        for index, record in enumerate(self.activity_log):
            key = (record["timestep"], record["player"].name)
            if decisions_by_key.get(key):
                decision = decisions_by_key[key].pop(0)
                events.append(self._decision_event(event_id, decision))
                event_id += 1

            events.append(self._action_event(event_id, record))
            event_id += 1
            snapshot = self._snapshot_for_action_index(index)
            events.append(self._state_event_from_snapshot(event_id, snapshot))
            event_id += 1

        return {
            "schema_version": "among-agents-game-log-v4",
            "game": {
                "seed": self.seed,
                "num_players": self.game_config["num_players"],
                "num_impostors": self.game_config["num_impostors"],
                "num_common_tasks": self.game_config["num_common_tasks"],
                "num_short_tasks": self.game_config["num_short_tasks"],
                "num_long_tasks": self.game_config["num_long_tasks"],
                "kill_cooldown": self.game_config["kill_cooldown"],
                "discussion_rounds": self.game_config["discussion_rounds"],
                "max_timesteps": self.game_config["max_timesteps"],
                "winner_code": winner,
                "winner": self._winner_text(winner),
                "timesteps": self.timestep,
                "task_progress": self.task_assignment.check_task_completion(),
            },
            "agents": {
                player_id(player): dict(self.agent_metadata.get(player.name, {}))
                for player in self.players
            },
            "experiment_config": self._experiment_config_payload(),
            "task_definitions": self._task_definitions_payload(),
            "task_assignments": self._task_assignments_payload(),
            "players": [
                {
                    "id": player_id(player),
                    "name": player.name,
                    "color": player.color,
                    "role": player.identity,
                    "initial_location": config["location"],
                    "agent_type": self.agent_metadata.get(player.name, {}).get(
                        "agent_type", "unknown"
                    ),
                    "model": self.agent_metadata.get(player.name, {}).get("model", ""),
                    "config": self._safe_config_payload(config),
                    "agent_log": self._agent_log(self.agents[index]),
                }
                for index, (player, config) in enumerate(
                    zip(self.players, self.resolved_player_configs)
                )
            ],
            "map": self._map_payload(),
            "events": events,
        }

    def _experiment_config_payload(self):
        return {
            "game_settings": dict(self.experiment_config.resolved_game_settings()),
            "seed": self.experiment_config.seed,
            "player_configs": [
                self._safe_config_payload(config)
                for config in self.resolved_player_configs
            ],
            "task_config": {
                name: dict(config)
                for name, config in self.experiment_config.task_config.items()
            },
            "map": self._map_payload(),
        }

    def _safe_config_payload(self, config):
        safe = {}
        for key, value in config.items():
            if key == "agent":
                safe[key] = type(value).__name__
            elif callable(value):
                safe[key] = getattr(value, "__name__", type(value).__name__)
            elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                safe[key] = value
            else:
                safe[key] = type(value).__name__
        return safe

    def _task_definitions_payload(self):
        by_name = {}
        for task in self.task_assignment.tasks:
            by_name.setdefault(
                task.name,
                {
                    "name": task.name,
                    "type": task.task_type,
                    "duration": task.max_duration,
                    "locations": [],
                },
            )
            by_name[task.name]["locations"].append(task.location)
        return list(by_name.values())

    def _task_assignments_payload(self):
        return {
            player_id(player): [
                {
                    "name": task.name,
                    "type": task.task_type,
                    "location": task.location,
                    "duration": task.max_duration,
                    "completed": task.check_completion(),
                }
                for task in player.tasks
            ]
            for player in self.players
        }

    def _map_payload(self):
        return {
            "rooms": {
                name: {
                    "coords": list(map_coords[name]["coords"])
                    if name in map_coords
                    else None,
                    "tasks": list(self.map.ship_map.nodes[name].get("tasks", [])),
                    "vents": list(self.map.get_adjacent_rooms_vent(name)),
                    "special_actions": list(
                        self.map.ship_map.nodes[name].get("special_actions", [])
                    ),
                }
                for name in self.map.ship_map.nodes
            },
            "corridors": [
                [room_a, room_b] for room_a, room_b in self.map.corridor_connections
            ],
            "vents": [
                [room_a, room_b] for room_a, room_b in self.map.vent_connections
            ],
        }

    def _state_snapshot(self, timestep, phase, summary, action_index):
        return {
            "timestep": timestep,
            "phase": phase,
            "summary": summary,
            "action_index": action_index,
            "players": self._player_states(),
            "task_progress": self.task_assignment.check_task_completion(),
            "meeting_info": dict(self.last_meeting_info or {}),
        }

    def _snapshot_for_action_index(self, action_index):
        for snapshot in self.state_log:
            if snapshot["action_index"] == action_index:
                return snapshot
        return self._state_snapshot(
            timestep=self.activity_log[action_index]["timestep"],
            phase=self.activity_log[action_index]["phase"],
            summary=f"State after action {action_index + 1}",
            action_index=action_index,
        )

    def _state_event_from_snapshot(self, event_id, snapshot):
        return {
            "event_id": event_id,
            "timestep": snapshot["timestep"],
            "phase": snapshot["phase"],
            "type": "state",
            "summary": snapshot["summary"],
            "players": snapshot["players"],
            "task_progress": snapshot["task_progress"],
            "meeting_info": snapshot.get("meeting_info", {}),
        }

    def _decision_event(self, event_id, decision):
        return {
            "event_id": event_id,
            "timestep": decision.get("timestep"),
            "phase": decision.get("phase"),
            "type": "agent_decision",
            "player_id": player_id(decision.get("player")),
            "private": True,
            "visible_to": [player_id(decision.get("player"))],
            "memory": decision.get("memory", ""),
            "thought": decision.get("thought", ""),
            "requested_action": decision.get("requested_action", ""),
            "chosen_action": decision.get("chosen_action", ""),
            "speech_message": decision.get("speech_message", ""),
            "agent_type": decision.get("agent_type", ""),
        }

    def _action_event(self, event_id, record):
        player = record["player"]
        action = record["action"]
        payload = {
            "event_id": event_id,
            "timestep": record["timestep"],
            "phase": record["phase"],
            "type": "action",
            "player_id": player_id(player),
            "action": self._action_payload(action),
            "public_text": self.message_system.create_action_message(record),
        }
        if "round" in record:
            payload["round"] = record["round"]
        return payload

    def _player_states(self):
        return [
            {
                "id": player_id(player),
                "location": player.location,
                "alive": player.is_alive,
                "reported_death": player.reported_death,
            }
            for player in self.players
        ]

    def _action_payload(self, action):
        payload = {"kind": action.name, "text": repr(action)}
        if isinstance(action, MoveTo):
            payload.update({"from": action.current_location, "to": action.new_location})
        if isinstance(action, Vent):
            payload.update({"from": action.current_location, "to": action.new_location})
        if isinstance(action, Kill):
            payload["target_player_id"] = player_id(action.other_player)
        if isinstance(action, Vote):
            payload["target_player_id"] = player_id(action.other_player)
        if isinstance(action, Speak):
            payload["message"] = action.message
        if isinstance(action, (CompleteTask, CompleteFakeTask)):
            payload["task"] = {
                "name": action.task.name,
                "location": action.task.location,
                "type": action.task.task_type,
                "completed": action.task.check_completion(),
            }
        if isinstance(action, CallMeeting):
            payload["location"] = action.current_location
            payload["meeting_info"] = dict(self.last_meeting_info or {})
        if isinstance(action, ViewMonitor):
            payload["location"] = action.current_location
        return payload

    @staticmethod
    def _safe_action_record(record):
        return {
            key: repr(value) if key == "action" else value
            for key, value in record.items()
        }

    @staticmethod
    def _agent_type_name(agent) -> str:
        if callable(getattr(agent, "agent_name", None)):
            return str(agent.agent_name())
        return type(agent).__name__

    @staticmethod
    def _model_name(agent) -> str:
        if callable(getattr(agent, "model_name", None)):
            return str(agent.model_name())
        return ""

    @staticmethod
    def _agent_log(agent):
        if callable(getattr(agent, "get_log", None)):
            return agent.get_log()
        return None

    @staticmethod
    def _winner_text(winner):
        return {
            1: "Impostors win: crewmates outnumbered or tied",
            2: "Crewmates win: impostors eliminated",
            3: "Crewmates win: all tasks completed",
            4: "Impostors win: time limit reached",
        }.get(winner, "unknown")


def run_amongus_game(*, game_config=None, UI=None):
    game = ConfiguredAmongUs(
        game_config=game_config,
        UI=UI,
    )
    winner = game.run_game()
    return game.result_summary(winner)
