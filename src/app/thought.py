from enum import Enum
import time
from typing import Any, Dict, List, NamedTuple, Optional

from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander


CHECKMARK_EMOJI = "✅"
THINKING_EMOJI = ":thinking_face:"
HISTORY_EMOJI = ":books:"
EXCEPTION_EMOJI = "⚠️"


def _convert_newlines(text: str) -> str:
    """Convert newline characters to markdown newline sequences
    (space, space, newline).
    """
    return text.replace("\n", "  \n")


class LLMThoughtState(Enum):
    """Enumerator of the LLMThought state."""

    # The LLM is thinking about what to do next. We don't know which tool we'll run.
    THINKING = "THINKING"
    # The LLM has decided to run a tool. We don't have results from the tool yet.
    RUNNING_TOOL = "RUNNING_TOOL"
    # We have results from the tool.
    COMPLETE = "COMPLETE"


class ToolRecord(NamedTuple):
    """Tool record as a NamedTuple."""

    name: str
    input_str: str


class LLMThoughtLabeler:
    """
    Generates markdown labels for LLMThought containers. Pass a custom
    subclass of this to StreamlitCallbackHandler to override its default
    labeling logic.
    """

    @staticmethod
    def get_initial_label() -> str:
        """Return the markdown label for a new LLMThought that doesn't have
        an associated tool yet.
        """
        return f"{THINKING_EMOJI} **Thinking...**"

    @staticmethod
    def get_tool_label(input_str, name, is_complete: bool) -> str:
        """Return the label for an LLMThought that has an associated
        tool.

        Parameters
        ----------
        tool
            The tool's ToolRecord

        is_complete
            True if the thought is complete; False if the thought
            is still receiving input.

        Returns
        -------
        The markdown label for the thought's container.

        """
        input = input_str
        emoji = CHECKMARK_EMOJI if is_complete else THINKING_EMOJI
        if name == "_Exception":
            emoji = EXCEPTION_EMOJI
            name = "Parsing error"
        idx = min([60, len(input)])
        input = input[0:idx]
        if len(input_str) > idx:
            input = input + "..."
        input = input.replace("\n", " ")
        label = f"{emoji} **{name}:** {input}"
        return label

    @staticmethod
    def get_history_label() -> str:
        """Return a markdown label for the special 'history' container
        that contains overflow thoughts.
        """
        return f"{HISTORY_EMOJI} **History**"

    @staticmethod
    def get_final_agent_thought_label() -> str:
        """Return the markdown label for the agent's final thought -
        the "Now I have the answer" thought, that doesn't involve
        a tool.
        """
        return f"{CHECKMARK_EMOJI} **Complete!**"


class LLMThought:
    """A thought in the LLM's thought stream."""

    def __init__(
        self,
        parent_container: Any,
        labeler: LLMThoughtLabeler = LLMThoughtLabeler(),
        expanded: bool = True,
        collapse_on_complete: bool = True,
    ):
        """Initialize the LLMThought.

        Args:
            parent_container: The container we're writing into.
            labeler: The labeler to use for this thought.
            expanded: Whether the thought should be expanded by default.
            collapse_on_complete: Whether the thought should be collapsed.
        """
        self._container = MutableExpander(
            parent_container=parent_container,
            label=labeler.get_initial_label(),
            expanded=expanded,
        )
        self._state = LLMThoughtState.THINKING
        self._llm_token_stream = ""
        self._llm_token_writer_idx: Optional[int] = None
        self._collapse_on_complete = collapse_on_complete
        self._labeler = labeler

    @property
    def container(self) -> MutableExpander:
        """The container we're writing into."""
        return self._container

    def _reset_llm_token_stream(self) -> None:
        self._llm_token_stream = ""
        self._llm_token_writer_idx = None

    def on_llm_start(self) -> None:
        self._reset_llm_token_stream()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # This is only called when the LLM is initialized with `streaming=True`
        # self._llm_token_stream += _convert_newlines(token)
        self._llm_token_stream += token.page_content
        self._llm_token_writer_idx = self._container.markdown(
            self._llm_token_stream, index=self._llm_token_writer_idx
        )

    def on_llm_end(self, **kwargs: Any) -> None:
        # `response` is the concatenation of all the tokens received by the LLM.
        # If we're receiving streaming tokens from `on_llm_new_token`, this response
        # data is redundant
        self._reset_llm_token_stream()

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._container.markdown("**LLM encountered an error...**")
        self._container.exception(error)

    def on_tool_start(self, input_str: str, tool_name: str, **kwargs: Any) -> None:
        # Called with the name of the tool we're about to run (in `serialized[name]`),
        # and its input. We change our container's label to be the tool name.
        self._state = LLMThoughtState.RUNNING_TOOL
        self._container.update(
            new_label=self._labeler.get_tool_label(
                input_str, tool_name, is_complete=False
            )
        )

    def on_tool_end(
        self,
        output: Any,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._container.markdown(f"**{str(output)}**")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self._container.markdown("**Tool encountered an error...**")
        self._container.exception(error)

    def complete(self, final_label: Optional[str] = None) -> None:
        """Finish the thought."""
        if final_label is None and self._state == LLMThoughtState.RUNNING_TOOL:
            assert (
                self._last_tool is not None
            ), "_last_tool should never be null when _state == RUNNING_TOOL"
            final_label = self._labeler.get_tool_label(
                self._last_tool, is_complete=True
            )
        self._state = LLMThoughtState.COMPLETE
        if self._collapse_on_complete:
            self._container.update(new_label=final_label, new_expanded=False)
        else:
            self._container.update(new_label=final_label)

    def clear(self) -> None:
        """Remove the thought from the screen. A cleared thought can't be reused."""
        self._container.clear()

    def search(self, doc):
        self.on_tool_start("Finding from vector database", "ChromaDB")
        # self.on_tool_end(doc)
        self._container.markdown(doc.page_content)
        self._container.markdown(doc.metadata)
        self._state = LLMThoughtState.COMPLETE
        self.complete("Completed")
