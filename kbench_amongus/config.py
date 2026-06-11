from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kbench_amongus.envs.configs.game_config import FIVE_MEMBER_GAME
from kbench_amongus.envs.configs.map_config import (
    connections as DEFAULT_CONNECTIONS,
    room_data as DEFAULT_ROOM_DATA,
    vent_connections as DEFAULT_VENT_CONNECTIONS,
)
from kbench_amongus.envs.configs.task_config import task_config as DEFAULT_TASK_CONFIG


@dataclass
class GameConfig:
    """Experiment config for an explicit-player Among Us run."""

    game_settings: dict[str, Any] = field(
        default_factory=lambda: {
            **FIVE_MEMBER_GAME,
            "num_players": 5,
            "num_impostors": 1,
        }
    )
    player_configs: list[dict[str, Any]] = field(default_factory=list)
    task_config: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            name: dict(config) for name, config in DEFAULT_TASK_CONFIG.items()
        }
    )
    map_room_data: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            name: {
                "tasks": list(details["tasks"]),
                "vent": list(details["vent"]),
                "special_actions": list(details["special_actions"]),
                "players": [],
            }
            for name, details in DEFAULT_ROOM_DATA.items()
        }
    )
    map_connections: list[tuple[str, str]] = field(
        default_factory=lambda: [tuple(edge) for edge in DEFAULT_CONNECTIONS]
    )
    map_vent_connections: list[tuple[str, str]] = field(
        default_factory=lambda: [tuple(edge) for edge in DEFAULT_VENT_CONNECTIONS]
    )
    seed: int = 7

    def resolved_game_settings(self) -> dict[str, Any]:
        settings = {
            **FIVE_MEMBER_GAME,
            "num_players": 5,
            "num_impostors": 1,
            **self.game_settings,
        }
        if self.player_configs:
            settings["num_players"] = len(self.player_configs)
            settings["num_impostors"] = sum(
                1
                for player in self.player_configs
                if str(player.get("role", "crewmate")).lower() in {"impostor", "imposter"}
            )
        return settings

    def with_updates(self, **updates) -> "GameConfig":
        data = {
            "game_settings": dict(self.game_settings),
            "player_configs": list(self.player_configs),
            "task_config": {
                name: dict(config) for name, config in self.task_config.items()
            },
            "map_room_data": {
                name: {
                    "tasks": list(details["tasks"]),
                    "vent": list(details["vent"]),
                    "special_actions": list(details["special_actions"]),
                    "players": [],
                }
                for name, details in self.map_room_data.items()
            },
            "map_connections": [tuple(edge) for edge in self.map_connections],
            "map_vent_connections": [tuple(edge) for edge in self.map_vent_connections],
            "seed": self.seed,
        }
        data.update(updates)
        return GameConfig(**data)
