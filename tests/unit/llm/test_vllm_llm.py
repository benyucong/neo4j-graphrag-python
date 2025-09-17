#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from unittest.mock import MagicMock, Mock, patch

import openai
import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.llm.vllm import VLLMLLM


def get_mock_openai() -> MagicMock:
    mock = MagicMock()
    mock.OpenAIError = openai.OpenAIError
    return mock


@patch("builtins.__import__", side_effect=ImportError)
def test_vllm_llm_missing_dependency(mock_import: Mock) -> None:
    with pytest.raises(ImportError):
        VLLMLLM(model_name="mistral")


@patch("builtins.__import__")
def test_vllm_llm_happy_path(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="vllm chat response"))],
    )

    llm = VLLMLLM(model_name="gpt", base_url="http://localhost:8000/v1", api_key="EMPTY")

    res = llm.invoke("my text")
    assert isinstance(res, LLMResponse)
    assert res.content == "vllm chat response"

    # Verify call args
    llm.client.chat.completions.create.assert_called_once()  # type: ignore
    call_kwargs = llm.client.chat.completions.create.call_args[1]  # type: ignore
    assert call_kwargs["model"] == "gpt"
    assert call_kwargs["messages"] == [{"role": "user", "content": "my text"}]


@patch("builtins.__import__")
def test_vllm_llm_with_message_history_and_system_instruction(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="vllm chat response"))],
    )

    llm = VLLMLLM(model_name="gpt", base_url="http://localhost:8000/v1", api_key="EMPTY")

    system_instruction = "You are a helpful assistant."
    message_history = [
        {"role": "user", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]
    question = "What about next season?"

    res = llm.invoke(question, message_history, system_instruction=system_instruction)  # type: ignore
    assert isinstance(res, LLMResponse)
    assert res.content == "vllm chat response"

    # Expected messages order: system, history..., user(question)
    expected_messages = [{"role": "system", "content": system_instruction}]
    expected_messages.extend(message_history)
    expected_messages.append({"role": "user", "content": question})

    llm.client.chat.completions.create.assert_called_once()  # type: ignore
    call_kwargs = llm.client.chat.completions.create.call_args[1]  # type: ignore
    assert call_kwargs["messages"] == expected_messages
    assert call_kwargs["model"] == "gpt"


@patch("builtins.__import__")
def test_vllm_llm_with_message_history_validation_error(mock_import: Mock) -> None:
    mock_openai = get_mock_openai()
    mock_import.return_value = mock_openai
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="vllm chat response"))],
    )

    llm = VLLMLLM(model_name="gpt", base_url="http://localhost:8000/v1", api_key="EMPTY")
    bad_history = [
        {"role": "human", "content": "When does the sun come up in the summer?"},
        {"role": "assistant", "content": "Usually around 6am."},
    ]

    with pytest.raises(LLMGenerationError) as exc_info:
        llm.invoke("Next season?", bad_history)  # type: ignore
    assert "Input should be 'user', 'assistant' or 'system'" in str(exc_info.value)
