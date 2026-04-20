"""Default system prompt for the agent.

The prompt is built at import time from scikit-rec's factory enum maps, so
whenever scikit-rec adds a new recommender/scorer/estimator, the prompt picks
it up without manual sync.
"""

from scikit_rec_agent.prompts.system import DEFAULT_SYSTEM_PROMPT

__all__ = ["DEFAULT_SYSTEM_PROMPT"]
