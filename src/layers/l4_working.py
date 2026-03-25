"""L4 Working Memory: Per-user session state management."""

from __future__ import annotations

import logging
import time

from src.models import L4State

logger = logging.getLogger(__name__)

# Emotion decay rate per turn (intensity *= decay)
EMOTION_DECAY = 0.7
# Minimum intensity before resetting to neutral
EMOTION_MIN_THRESHOLD = 0.15
# Maximum conversation turns to track
MAX_TOPICS = 5


class L4WorkingMemory:
    """In-memory per-user session state. No API calls, pure rule-based updates."""

    def __init__(self):
        self._sessions: dict[str, L4State] = {}
        self._last_active: dict[str, float] = {}
        # Session timeout in seconds (30 minutes)
        self._timeout = 1800

    def get_state(self, user_id: str) -> L4State:
        """Get or create session state for a user."""
        self._cleanup_stale()

        if user_id not in self._sessions:
            self._sessions[user_id] = L4State()
            self._last_active[user_id] = time.time()

        return self._sessions[user_id]

    def update(
        self,
        user_id: str,
        emotion_hint: str = "",
        triggers: list[str] | None = None,
        user_message: str = "",
    ) -> L4State:
        """Update session state based on current turn analysis.

        Args:
            user_id: User identifier.
            emotion_hint: Detected emotion from Step A.
            triggers: Situation tags from Step A.
            user_message: Current user message (for topic tracking).

        Returns:
            Updated L4State.
        """
        state = self.get_state(user_id)
        self._last_active[user_id] = time.time()

        # Increment turn counter
        state.conversation_turns += 1

        # Update emotion
        if emotion_hint:
            if emotion_hint == state.current_emotion:
                # Same emotion → intensify (but cap at 1.0)
                state.emotion_intensity = min(1.0, state.emotion_intensity + 0.2)
            else:
                # New emotion → switch
                state.current_emotion = emotion_hint
                state.emotion_intensity = 0.5
        else:
            # No emotion signal → decay
            state.emotion_intensity *= EMOTION_DECAY
            if state.emotion_intensity < EMOTION_MIN_THRESHOLD:
                state.current_emotion = "平静"
                state.emotion_intensity = 0.3

        # Update active topics from triggers
        if triggers:
            for tag in triggers:
                if tag in state.active_topics:
                    state.active_topics.remove(tag)
                state.active_topics.insert(0, tag)
            state.active_topics = state.active_topics[:MAX_TOPICS]

        # Update summary (simple: keep last message snippet)
        if user_message:
            state.summary = user_message[:100]

        # Relationship estimate based on turn count
        state.relationship_estimate = self._estimate_relationship(state.conversation_turns)

        return state

    def format_state(self, user_id: str) -> str:
        """Format L4 state as natural language for prompt injection (不含用户身份，由 user_prompts 提供)。"""
        state = self.get_state(user_id)

        parts = []

        # Current emotional state
        if state.current_emotion != "平静" and state.emotion_intensity > 0.3:
            intensity_word = "强烈" if state.emotion_intensity > 0.7 else "略微"
            parts.append(f"你现在{intensity_word}感到{state.current_emotion}")

        # Conversation context
        if state.conversation_turns > 1:
            parts.append(f"这是你们对话的第{state.conversation_turns}轮")

        # Active topics
        if state.active_topics:
            topics_str = "、".join(state.active_topics[:3])
            parts.append(f"当前话题涉及：{topics_str}")

        return "。".join(parts) + "。" if parts else ""

    def reset(self, user_id: str) -> None:
        """Reset a user's session state."""
        self._sessions.pop(user_id, None)
        self._last_active.pop(user_id, None)

    def _cleanup_stale(self) -> None:
        """Remove sessions that have been inactive past timeout."""
        now = time.time()
        stale = [
            uid for uid, last in self._last_active.items()
            if now - last > self._timeout
        ]
        for uid in stale:
            self._sessions.pop(uid, None)
            self._last_active.pop(uid, None)

    @staticmethod
    def _estimate_relationship(turns: int) -> str:
        """Simple heuristic for relationship based on conversation length."""
        if turns <= 2:
            return "陌生"
        if turns <= 10:
            return "初识"
        if turns <= 30:
            return "熟悉"
        return "亲近"
