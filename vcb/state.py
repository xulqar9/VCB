"""Explicit conversation state.

The pipeline's activity is spread over several independent signals (the turn
task, local playback, avatar playback, the mic's VAD). Rather than keep yet
another flag in sync with them, `PipelineSnapshot` captures those signals at
one instant and derives a single `Phase` plus the guard predicates the
orchestrator needs. Every guard goes through here, so adding a new signal means
changing one place instead of every hand-rolled boolean combination.
"""

from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    IDLE = "idle"              # nobody is talking, nothing in flight
    LISTENING = "listening"    # user is mid-utterance
    THINKING = "thinking"      # a turn is running but no audio is out yet
    SPEAKING = "speaking"      # assistant audio is playing (locally or on avatar)


@dataclass(frozen=True)
class PipelineSnapshot:
    user_speaking: bool = False
    turn_active: bool = False
    audio_playing: bool = False
    avatar_speaking: bool = False

    @property
    def assistant_active(self) -> bool:
        """Assistant is thinking or speaking -- anything a barge-in would cut."""
        return self.turn_active or self.audio_playing or self.avatar_speaking

    @property
    def busy(self) -> bool:
        """Anyone is talking or a reply is in flight; AI-led turns must wait."""
        return self.user_speaking or self.assistant_active

    @property
    def phase(self) -> Phase:
        # User speech wins: during a barge-in the old reply is being torn down
        # and the user's utterance is what matters.
        if self.user_speaking:
            return Phase.LISTENING
        if self.audio_playing or self.avatar_speaking:
            return Phase.SPEAKING
        if self.turn_active:
            return Phase.THINKING
        return Phase.IDLE
