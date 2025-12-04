"""
Next With Summary Tool - Handles memory update, chunking, and context reset.

This tool:
- Chunks the document from instance['context'] on first use
- Updates cumulative memory with provided summary
- Resets conversation history to system + problem + memory + next chunk
- Manages chunk index and provides feedback about progress
"""

from skyrl_agent.tools.base import BaseTool, register_tool
from skyrl_agent.tools.prompt import SIMPLE_MEMORY_SYSTEM_PROMPT
from typing import Union, List, Optional
import re


@register_tool("next_with_summary")
class NextWithSummary(BaseTool):
    name = "next_with_summary"
    description = "Update your memory with new information from the current section and move to the next chunk. Provide a summary that combines important details from previous memory with new findings from this section. This tool will reset the conversation context, keeping only your updated memory and the next section."
    parameters = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Updated memory that combines: (1) relevant information from previous memory, and (2) new information from the current section that helps answer the problem. Be comprehensive but concise.",
            }
        },
        "required": ["summary"],
    }

    def get_system_prompt_prefix(self) -> Optional[str]:
        """Return the system prompt prefix required for this tool."""
        return SIMPLE_MEMORY_SYSTEM_PROMPT

    def _chunk_documents_by_tokens(self, context: str, tokenizer, max_tokens: int = 4000) -> List[str]:
        """Chunk documents by tokens. Groups by 'Document X:' and packs up to max_tokens.

        Uses the provided tokenizer to count tokens per doc block.
        """
        doc_pattern = r"(Document \d+:)"
        parts = re.split(doc_pattern, context)

        documents: List[str] = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                doc = parts[i] + "\n" + parts[i + 1]
                documents.append(doc.strip())
            elif i < len(parts):
                documents.append(parts[i].strip())

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_tokens = 0

        for doc in documents:
            doc_tokens = len(tokenizer.encode(doc))
            if current_tokens + doc_tokens > max_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [doc]
                current_tokens = doc_tokens
            else:
                current_chunk.append(doc)
                current_tokens += doc_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _ensure_document_chunks(self, agent) -> None:
        """Initialize chunking from instance['context'] on first use.

        Also stores system prompt and problem for later context resets.
        """
        # Use agent attributes to store chunking state
        if hasattr(agent, "document_chunks") and agent.document_chunks:
            return

        raw_context = agent.instance.get("context")
        if not raw_context or not isinstance(raw_context, str):
            raise ValueError("next_with_summary requires instance['context'] (str) to be set for chunking.")

        # Chunk the documents
        agent.document_chunks = self._chunk_documents_by_tokens(
            raw_context,
            agent.tokenizer,
            4000,
        )
        agent.current_chunk_index = 0
        agent.cumulative_memory = ""

        # Store system prompt and problem for context resets
        # Extract system prompt
        for msg in agent.history.messages:
            if msg.get("role") == "system":
                agent.mem_system_prompt = msg.get("content", "")
                break

        # Extract problem from user message
        problem_content = ""
        for msg in agent.history.messages:
            if msg.get("role") == "user":
                problem_content = msg.get("content", "")
                break

        # Extract just the problem text if wrapped in tags
        if "<problem>" in problem_content:
            problem_match = re.search(r"<problem>(.*?)</problem>", problem_content, re.DOTALL)
            if problem_match:
                agent.mem_cur_question = problem_match.group(1).strip()
            else:
                agent.mem_cur_question = problem_content
        else:
            agent.mem_cur_question = problem_content

    def _build_flushed_user_message(self, agent, next_chunk: str) -> str:
        """Build the user message with problem, memory, next section, and feedback."""
        memory_section = (
            f"<memory>\n{agent.cumulative_memory}\n</memory>"
            if agent.cumulative_memory
            else "<memory>\nNo information recorded yet.\n</memory>"
        )

        section_content = (
            f"<section>\nChunk {agent.current_chunk_index}/{len(agent.document_chunks)}:\n\n{next_chunk}\n</section>"
        )

        # Use stored problem question
        problem_text = getattr(agent, "mem_cur_question", "")

        # Build feedback message about progress
        if agent.current_chunk_index >= len(agent.document_chunks):
            feedback = (
                f"Memory updated. Moved to chunk {agent.current_chunk_index}/{len(agent.document_chunks)} "
                "(FINAL CHUNK). After reading this chunk, use the finish tool to provide your answer."
            )
        else:
            feedback = (
                f"Memory updated. Moved to chunk {agent.current_chunk_index}/{len(agent.document_chunks)}. "
                "Continue reading and updating your memory."
            )

        return (
            f"<problem>\n{problem_text}\n</problem>\n\n"
            f"{memory_section}\n\n{section_content}\n\n"
            f"**Progress:** {feedback}"
        )

    def _flush_history_with(self, agent, user_content: str) -> None:
        """Reset history to only system + new user message.

        The history.reset flag will be set by initialize(), which will be
        detected in _prepare_llm_input() to reset prompt_token_len.
        """
        # Use stored system prompt
        system_content = getattr(agent, "mem_system_prompt", "")

        # Replace history with only system + new user message
        new_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        agent.history.initialize(new_messages)

    def call(self, params: dict, **kwargs) -> Optional[Union[str, dict]]:
        """Handle next_with_summary: update memory, get next chunk, reset context.

        Returns None to indicate that feedback is embedded in the user message
        and no tool output should be added to history.
        """
        agent = kwargs.get("agent")
        if agent is None:
            return "Error: Agent context missing for next_with_summary. Use MemAgent which passes agent=self."

        params = self._verify_json_format_args(params)

        # Get summary from params
        summary = params.get("summary", "")
        if not summary:
            return "Error: 'summary' parameter is required."

        # Ensure chunks are initialized
        self._ensure_document_chunks(agent)

        # Update cumulative memory with provided summary
        agent.cumulative_memory = summary

        # Check if we've run out of chunks
        if agent.current_chunk_index >= len(agent.document_chunks):
            return "You have read all chunks. Please use the finish tool to provide your final " "answer IMMEDIATELY."

        # Get next chunk and advance index
        next_chunk = agent.document_chunks[agent.current_chunk_index]
        agent.current_chunk_index += 1

        # Flush context with memory + next section (feedback is embedded in user message)
        new_user_message = self._build_flushed_user_message(agent, next_chunk)
        self._flush_history_with(agent, new_user_message)

        # Return None to indicate no tool output should be added to history
        # (feedback is already in the new user message)
        return None
