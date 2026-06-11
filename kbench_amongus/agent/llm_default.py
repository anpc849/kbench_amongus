from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from kbench_amongus.agent.base import BaseAgent
from kbench_amongus.agent.prompts import (
    CONNECTION_INFO,
    CREWMATE_EXAMPLE,
    CREWMATE_PROMPT,
    IMPOSTOR_EXAMPLE,
    IMPOSTOR_PROMPT,
    MEETING_PHASE_INSTRUCTION,
    TASK_PHASE_INSTRUCTION,
)
from kbench_amongus.agent.validation import InvalidAgentError
from kbench_amongus.envs.action import Speak, Vote


@dataclass
class LLMActionDecision:
    memory: str
    thought: str
    action: str
    speech_message: str = ""


class DefaultLLMAgent(BaseAgent):
    """Minimal game agent for a raw kaggle-benchmarks LLM object."""

    response_schema = LLMActionDecision

    def __init__(self, llm, max_retries: int = 2, llm_pause_seconds: float = 1.0):
        self.llm = llm
        self.max_retries = max_retries
        self.llm_pause_seconds = llm_pause_seconds
        self.memory = ""
        self.thought = ""
        self.decision_log: list[dict[str, Any]] = []

    def choose_action(self):
        context = self.game.build_agent_context(self.player)
        if not context.available_actions:
            return None
        retry_note = ""
        last_error = None
        decision = None
        action = None
        for _ in range(self.max_retries + 1):
            decision = self._call_llm(context, retry_note=retry_note)
            try:
                action = self._decision_to_action(decision, context.available_actions)
            except InvalidAgentError as exc:
                last_error = exc
                retry_note = self._retry_note(exc, context.available_actions)
                continue
            break
        else:
            raise InvalidAgentError(
                f"{self.player.name} ({self.player.identity}) could not produce a "
                f"legal action during {context.phase} at timestep {context.timestep} "
                f"after {self.max_retries + 1} attempts. Last error: {last_error}"
            ) from last_error
        self.memory = decision.memory.strip()
        self.thought = decision.thought.strip()
        self._record_decision(decision, action)
        return action

    def choose_observation_location(self, map_nodes):
        locations = sorted(list(map_nodes))
        if not locations:
            return "Cafeteria"
        context = self.game.build_agent_context(self.player)
        prompt = (
            f"{context.to_text()}\n\n"
            "You are using the security monitor. Choose exactly one room from "
            f"this list: {', '.join(locations)}. Return only the room name."
        )
        try:
            location = str(self.llm.prompt(prompt, schema=str)).strip()
            self._sleep_after_llm_call()
        except Exception as exc:
            raise InvalidAgentError(
                f"{self.player.name} ({self.player.identity}) monitor LLM call "
                f"failed: {exc}"
            ) from exc
        return location

    def get_log(self) -> dict[str, Any] | None:
        return {"memory": self.memory, "thought": self.thought}

    def agent_name(self) -> str:
        return type(self).__name__

    def model_name(self) -> str:
        for attr in ("model", "name", "id"):
            value = getattr(self.llm, attr, None)
            if value:
                return str(value)
        return type(self.llm).__name__

    def _call_llm(self, context, retry_note: str = ""):
        prompt = (
            self._system_text(context)
            + "\n\nMap guidance:\n"
            + CONNECTION_INFO
            + "\n\n"
            + context.to_text()
            + "\n\nPrevious memory:\n"
            + (self.memory or "None")
            + "\n\nPrevious thought:\n"
            + (self.thought or "None")
            + "\n\n"
            + self._phase_action_instruction(context)
            + "\n\nNow, write your response with memory, thought, and action. "
            "Make sure action is chosen from the available actions. Case sensitive. "
            "Return JSON with fields memory, thought, action, and speech_message. "
            "The action field must exactly match one listed action, except SPEAK "
            "may use action='SPEAK' with speech_message containing the utterance."
            + retry_note
        )
        try:
            decision = self.llm.prompt(prompt, schema=self.response_schema)
            self._sleep_after_llm_call()
            return decision
        except Exception as exc:
            raise InvalidAgentError(
                f"{self.player.name} ({self.player.identity}) LLM action call "
                f"failed during {context.phase} at timestep {context.timestep}: {exc}"
            ) from exc

    def _sleep_after_llm_call(self):
        if self.llm_pause_seconds > 0:
            time.sleep(self.llm_pause_seconds)

    def _system_text(self, context) -> str:
        if context.role == "Impostor":
            role_prompt = IMPOSTOR_PROMPT.format(name=context.player_name)
            example = IMPOSTOR_EXAMPLE
        else:
            role_prompt = CREWMATE_PROMPT.format(name=context.player_name)
            example = CREWMATE_EXAMPLE
        phase_instruction = (
            MEETING_PHASE_INSTRUCTION
            if context.phase == "meeting"
            else TASK_PHASE_INSTRUCTION
        )
        return (
            f"{role_prompt}\n"
            f"{phase_instruction}\n"
            f"{example}"
        )

    @staticmethod
    def _phase_action_instruction(context):
        available_actions = list(context.available_actions)
        if available_actions and all(isinstance(action, Vote) for action in available_actions):
            return (
                "Voting is active now. Discussion is over. SPEAK is not legal. "
                "You must choose exactly one listed VOTE action."
            )
        if available_actions and all(isinstance(action, Speak) for action in available_actions):
            return (
                "Discussion is active now. Voting is not open yet. Choose SPEAK "
                "with a non-empty speech_message."
            )
        return "Choose only from the currently listed legal actions."

    def _decision_to_action(self, decision, available_actions):
        requested = str(getattr(decision, "action", "")).strip()
        speech = str(getattr(decision, "speech_message", "")).strip()
        requested_normalized = self._normalize_action_text(requested)
        for action in available_actions:
            if (
                requested == repr(action)
                or requested_normalized == self._normalize_action_text(repr(action))
            ):
                return action
        for action in available_actions:
            if isinstance(action, Speak) and requested_normalized.startswith("speak"):
                action.provide_message(speech or self._extract_speech(requested))
                return action
        available = "\n".join(f"- {repr(action)}" for action in available_actions)
        raise InvalidAgentError(
            f"{self.player.name} ({self.player.identity}) requested illegal LLM "
            f"action {requested!r}. Legal actions are:\n{available}"
        )

    @staticmethod
    def _normalize_action_text(text):
        return " ".join(str(text).replace("_", " ").lower().split())

    @staticmethod
    def _retry_note(exc, available_actions):
        available = "\n".join(f"- {repr(action)}" for action in available_actions)
        if available_actions and all(isinstance(action, Vote) for action in available_actions):
            phase_note = (
                "Voting is active and discussion is closed. SPEAK is illegal now. "
                "Pick one exact VOTE action from the list."
            )
        elif available_actions and all(isinstance(action, Speak) for action in available_actions):
            phase_note = (
                "Discussion is active and voting is not open. Pick SPEAK and include "
                "a non-empty speech_message."
            )
        else:
            phase_note = "Do not choose an action unless it is listed above."
        return (
            "\n\nYour previous action was invalid.\n"
            f"Error: {exc}\n\n"
            "Choose exactly one action from this legal list now:\n"
            f"{available}\n\n"
            f"{phase_note}"
        )

    @staticmethod
    def _extract_speech(action_text):
        if ":" not in action_text:
            return ""
        return action_text.split(":", 1)[1].strip().strip("\"'")

    def _record_decision(self, decision, action):
        self.decision_log.append(
            {
                "player": self.player.name,
                "identity": self.player.identity,
                "timestep": None,
                "phase": None,
                "memory": str(getattr(decision, "memory", "")).strip(),
                "thought": str(getattr(decision, "thought", "")).strip(),
                "requested_action": str(getattr(decision, "action", "")).strip(),
                "speech_message": str(
                    getattr(decision, "speech_message", "")
                ).strip(),
                "chosen_action": repr(action),
                "agent_type": type(self).__name__,
            }
        )
