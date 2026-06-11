from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentContext:
    """Private, structured game context for one player at one decision point."""

    player_name: str
    role: str
    phase: str
    timestep: int
    location: str
    alive: bool
    task_progress: float
    game_settings: dict[str, Any]
    available_actions: list[Any]
    observations: list[str]
    action_history: list[dict[str, Any]]
    assigned_tasks: list[dict[str, Any]]
    visible_players: list[dict[str, Any]]
    meeting: dict[str, Any] | None
    map: dict[str, Any]

    @property
    def available_action_texts(self) -> list[str]:
        return [repr(action) for action in self.available_actions]

    def available_actions_text(self) -> str:
        return "\n".join(f"- {text}" for text in self.available_action_texts)

    def to_text(self) -> str:
        tasks = "\n".join(
            (
                f"- {task['name']} at {task['location']} "
                f"({task['type']}, completed={task['completed']})"
                + (f"; path={' -> '.join(task['path'])}" if task.get("path") else "")
            )
            for task in self.assigned_tasks
        ) or "- None"
        observations = "\n".join(f"- {item}" for item in self.observations) or "- None"
        action_history = "\n".join(
            f"- Timestep {record['timestep']}: [{record['phase']}] {record['action']}"
            for record in self.action_history
        ) or "- None"
        visible_players = "\n".join(
            (
                f"- {player['name']} at {player['location']} "
                f"(alive={player['alive']})"
            )
            for player in self.visible_players
        ) or "- None"
        meeting = self.meeting or {}
        meeting_info = meeting.get("meeting_info") or {}
        known_dead_players = ", ".join(
            (
                f"{player['name']} at {player['location']}"
                + (" (newly known)" if player.get("newly_known") else "")
            )
            for player in meeting_info.get("known_dead_players", [])
        ) or "None"
        newly_known_dead_players = ", ".join(
            f"{player['name']} at {player['location']}"
            for player in meeting_info.get("newly_known_dead_players", [])
        ) or "None"
        meeting_text = (
            "No active meeting."
            if not meeting
            else (
                f"round={meeting['round']}, rounds_left={meeting['rounds_left']}, "
                f"votes_so_far={meeting['votes_so_far']}\n"
                f"Meeting called by: {meeting_info.get('called_by', 'unknown')}\n"
                f"Meeting trigger: {meeting_info.get('trigger', 'unknown')}\n"
                f"Newly known dead players: {newly_known_dead_players}\n"
                f"Known dead players: {known_dead_players}"
            )
        )
        return (
            f"Player: {self.player_name}\n"
            f"Role: {self.role}\n"
            f"Phase: {self.phase}\n"
            f"Time: {self.timestep}/{self.game_settings['max_timesteps']}\n"
            f"Location: {self.location}\n"
            f"Global task progress: {self.task_progress * 100:.1f}%\n"
            f"Meeting: {meeting_text}\n\n"
            f"Visible players:\n{visible_players}\n\n"
            f"Recent observations:\n{observations}\n\n"
            f"Recent self action history:\n{action_history}\n\n"
            f"Assigned tasks:\n{tasks}\n\n"
            f"Available actions:\n{self.available_actions_text()}"
        )


class BaseAgent:
    """Base class for game-compatible player agents.

    User agents may subclass this class, but validation is behavioral: any
    object with the required methods can be used.
    """

    def bind(self, player, game):
        self.player = player
        self.game = game
        return self

    def setup(self) -> None:
        return None

    def choose_action(self):
        raise NotImplementedError("Agents must implement choose_action().")

    def choose_observation_location(self, map_nodes):
        locations = sorted(list(map_nodes))
        return locations[0] if locations else "Cafeteria"

    def get_log(self) -> dict[str, Any] | None:
        return None

    def agent_name(self) -> str:
        return type(self).__name__

    def model_name(self) -> str:
        return ""
