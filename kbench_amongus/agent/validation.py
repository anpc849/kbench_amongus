from __future__ import annotations

from typing import Any

from kbench_amongus.agent.base import BaseAgent
from kbench_amongus.envs.action import Speak


class InvalidAgentError(Exception):
    """Raised when a player agent cannot safely interact with the game."""


def is_raw_kbench_llm(value: Any) -> bool:
    if value is None:
        return False
    if callable(getattr(value, "bind", None)):
        return False
    return callable(getattr(value, "prompt", None)) and callable(
        getattr(value, "respond", None)
    )


def validate_agent_shape(agent: Any, *, player_name: str, role: str) -> None:
    if agent is None:
        raise InvalidAgentError(
            f"{player_name} ({role}) has no agent. "
            "player_configs entries must include {'role': ..., 'agent': ...}."
        )
    for method in ("bind", "choose_action", "choose_observation_location"):
        if not callable(getattr(agent, method, None)):
            raise InvalidAgentError(
                f"{player_name} ({role}) agent {type(agent).__name__} must define "
                f"callable {method}()."
            )
    if getattr(type(agent), "choose_action", None) is BaseAgent.choose_action:
        raise InvalidAgentError(
            f"{player_name} ({role}) agent {type(agent).__name__} must implement "
            "choose_action(); BaseAgent.choose_action() is abstract."
        )


def bind_and_validate_agent(agent: Any, player, game):
    validate_agent_shape(agent, player_name=player.name, role=player.identity)
    try:
        bound_agent = agent.bind(player, game)
    except Exception as exc:
        raise InvalidAgentError(
            f"{player.name} ({player.identity}) agent {type(agent).__name__}.bind() "
            f"failed: {exc}"
        ) from exc
    if bound_agent is None:
        raise InvalidAgentError(
            f"{player.name} ({player.identity}) agent {type(agent).__name__}.bind() "
            "returned None; it must return the bound agent."
        )
    validate_agent_shape(bound_agent, player_name=player.name, role=player.identity)
    if getattr(bound_agent, "player", None) is not player:
        raise InvalidAgentError(
            f"{player.name} ({player.identity}) agent {type(bound_agent).__name__} "
            "must set self.player to the bound player."
        )
    if callable(getattr(bound_agent, "setup", None)):
        try:
            bound_agent.setup()
        except Exception as exc:
            raise InvalidAgentError(
                f"{player.name} ({player.identity}) agent "
                f"{type(bound_agent).__name__}.setup() failed: {exc}"
            ) from exc
    return bound_agent


def validate_agent_action(agent, action, available_actions, *, phase: str, timestep: int):
    player = getattr(agent, "player", None)
    player_name = getattr(player, "name", "unknown player")
    role = getattr(player, "identity", "unknown role")
    available_actions = list(available_actions)
    if action is None:
        return None

    if action in available_actions:
        return _validate_speak_action(action, player_name, role, phase, timestep)

    action_text = action if isinstance(action, str) else repr(action)
    for candidate in available_actions:
        if repr(candidate) == action_text:
            return _validate_speak_action(candidate, player_name, role, phase, timestep)

    if isinstance(action, Speak) or action_text.startswith("SPEAK"):
        speak = _find_speak_action(available_actions)
        if speak is not None:
            message = getattr(action, "message", "") if isinstance(action, Speak) else ""
            if isinstance(action, str) and ":" in action:
                message = action.split(":", 1)[1].strip()
            speak.provide_message(message)
            return _validate_speak_action(speak, player_name, role, phase, timestep)

    available = "\n".join(f"- {repr(candidate)}" for candidate in available_actions)
    raise InvalidAgentError(
        f"{player_name} ({role}) returned illegal action at timestep {timestep} "
        f"during {phase}: {action_text!r}. Legal actions are:\n{available}"
    )


def validate_observation_location(agent, location: str, map_nodes):
    player = getattr(agent, "player", None)
    player_name = getattr(player, "name", "unknown player")
    role = getattr(player, "identity", "unknown role")
    valid_locations = sorted(list(map_nodes))
    if location in valid_locations:
        return location
    raise InvalidAgentError(
        f"{player_name} ({role}) returned invalid monitor location {location!r}. "
        f"Valid rooms are: {', '.join(valid_locations)}"
    )


def _find_speak_action(available_actions):
    for action in available_actions:
        if isinstance(action, Speak):
            return action
    return None


def _validate_speak_action(action, player_name: str, role: str, phase: str, timestep: int):
    if isinstance(action, Speak):
        message = str(getattr(action, "message", "")).strip()
        if not message or message == "...":
            raise InvalidAgentError(
                f"{player_name} ({role}) returned SPEAK without a non-empty "
                f"message at timestep {timestep} during {phase}."
            )
    return action
