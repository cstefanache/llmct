"""Unit tests for scenario module.

Tests cover model validation, field constraints, edge cases (empty lists,
None values, boundary values), and scenario loading via mocked file I/O.
"""
from __future__ import annotations

from unittest.mock import mock_open, patch

import pytest
from pydantic import ValidationError

from activation_lab.scenario import (
    CaptureConfig,
    GenerationConfig,
    Message,
    ModelConfig,
    OutputConfig,
    PromptConfig,
    ReferenceState,
    Scenario,
    load_scenario,
)


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_message_valid_roles(role: str) -> None:
    """Verify that all allowed role literals are accepted by Message."""
    msg = Message(role=role, content="hello")
    assert msg.role == role
    assert msg.content == "hello"


@pytest.mark.parametrize("bad_role", ["invalid", "bot", "human", "", "USER"])
def test_message_invalid_role(bad_role: str) -> None:
    """Verify that roles outside the allowed literal set raise ValidationError."""
    with pytest.raises(ValidationError):
        Message(role=bad_role, content="test")


def test_message_empty_content() -> None:
    """Verify that an empty string is a valid content value for Message."""
    msg = Message(role="user", content="")
    assert msg.content == ""


# ---------------------------------------------------------------------------
# PromptConfig
# ---------------------------------------------------------------------------


def test_prompt_config_single_message() -> None:
    """Verify PromptConfig accepts a list with a single Message."""
    prompt = PromptConfig(messages=[Message(role="user", content="test")])
    assert len(prompt.messages) == 1
    assert prompt.run_at_each_message is False


def test_prompt_config_multiple_messages() -> None:
    """Verify PromptConfig accepts multiple messages and preserves order."""
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]
    prompt = PromptConfig(messages=messages)
    assert len(prompt.messages) == 3
    assert prompt.messages[0].role == "system"


def test_prompt_config_run_at_each_message_flag() -> None:
    """Verify the run_at_each_message flag can be set to True."""
    prompt = PromptConfig(
        messages=[Message(role="user", content="x")],
        run_at_each_message=True,
    )
    assert prompt.run_at_each_message is True


def test_prompt_config_empty_messages() -> None:
    """Verify PromptConfig accepts an empty messages list (no constraint enforced)."""
    prompt = PromptConfig(messages=[])
    assert prompt.messages == []


# ---------------------------------------------------------------------------
# ReferenceState
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_label",
    [
        "test label",   # space
        "test/label",   # forward slash
        "test\\label",  # backslash
        "test\tlabel",  # tab
        "test\nlabel",  # newline
        "",             # empty string
    ],
)
def test_reference_state_invalid_labels(bad_label: str) -> None:
    """Verify that labels containing whitespace, slashes, or empty strings are rejected."""
    with pytest.raises(ValidationError):
        ReferenceState(label=bad_label, messages=[])


@pytest.mark.parametrize(
    "good_label",
    ["test_label", "label-1", "CamelCase", "label123", "a"],
)
def test_reference_state_valid_labels(good_label: str) -> None:
    """Verify that labels without whitespace or slashes are accepted."""
    ref = ReferenceState(label=good_label, messages=[])
    assert ref.label == good_label


def test_reference_state_empty_messages() -> None:
    """Verify ReferenceState accepts an empty messages list."""
    ref = ReferenceState(label="valid", messages=[])
    assert ref.messages == []


def test_reference_state_with_messages() -> None:
    """Verify ReferenceState stores messages correctly."""
    ref = ReferenceState(
        label="state1",
        messages=[Message(role="user", content="hello")],
    )
    assert len(ref.messages) == 1
    assert ref.messages[0].content == "hello"


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_model_config_valid_dtypes(dtype: str) -> None:
    """Verify all allowed dtype literals are accepted by ModelConfig."""
    model = ModelConfig(id="test_model", dtype=dtype)
    assert model.dtype == dtype


@pytest.mark.parametrize("bad_dtype", ["int8", "float64", "half", "", "FLOAT16"])
def test_model_config_invalid_dtype(bad_dtype: str) -> None:
    """Verify that dtype values outside the allowed set raise ValidationError."""
    with pytest.raises(ValidationError):
        ModelConfig(id="test", dtype=bad_dtype)


@pytest.mark.parametrize("device", ["auto", "mps", "cpu"])
def test_model_config_valid_devices(device: str) -> None:
    """Verify all allowed device literals are accepted by ModelConfig."""
    model = ModelConfig(id="test_model", device=device)
    assert model.device == device


def test_model_config_defaults() -> None:
    """Verify ModelConfig default values are set correctly."""
    model = ModelConfig(id="my_model")
    assert model.id == "my_model"
    assert model.dtype == "float16"
    assert model.device == "auto"
    assert model.trust_remote_code is False
    assert model.gguf_file is None


def test_model_config_gguf_file_none() -> None:
    """Verify gguf_file accepts None explicitly."""
    model = ModelConfig(id="test", gguf_file=None)
    assert model.gguf_file is None


def test_model_config_gguf_file_set() -> None:
    """Verify gguf_file accepts a string path."""
    model = ModelConfig(id="test", gguf_file="model.gguf")
    assert model.gguf_file == "model.gguf"


# ---------------------------------------------------------------------------
# GenerationConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_tokens", [0, -1, -100])
def test_generation_config_invalid_max_new_tokens(bad_tokens: int) -> None:
    """Verify max_new_tokens values below 1 raise ValidationError."""
    with pytest.raises(ValidationError):
        GenerationConfig(max_new_tokens=bad_tokens)


@pytest.mark.parametrize("good_tokens", [1, 5, 100, 4096])
def test_generation_config_valid_max_new_tokens(good_tokens: int) -> None:
    """Verify max_new_tokens accepts values >= 1."""
    gen = GenerationConfig(max_new_tokens=good_tokens)
    assert gen.max_new_tokens == good_tokens


def test_generation_config_defaults() -> None:
    """Verify GenerationConfig default values are set correctly."""
    gen = GenerationConfig()
    assert gen.max_new_tokens == 8
    assert gen.do_sample is False
    assert gen.temperature == 1.0
    assert gen.top_k is None
    assert gen.top_p is None
    assert gen.seed == 42


def test_generation_config_optional_none_values() -> None:
    """Verify top_k and top_p accept None explicitly."""
    gen = GenerationConfig(top_k=None, top_p=None)
    assert gen.top_k is None
    assert gen.top_p is None


def test_generation_config_optional_set_values() -> None:
    """Verify top_k and top_p accept numeric values."""
    gen = GenerationConfig(top_k=50, top_p=0.9)
    assert gen.top_k == 50
    assert gen.top_p == 0.9


# ---------------------------------------------------------------------------
# CaptureConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_capture_config_valid_store_dtypes(dtype: str) -> None:
    """Verify all allowed store_dtype literals are accepted by CaptureConfig."""
    cap = CaptureConfig(store_dtype=dtype)
    assert cap.store_dtype == dtype


def test_capture_config_top_k_probs_zero() -> None:
    """Verify top_k_probs accepts 0 (boundary value, ge=0)."""
    cap = CaptureConfig(top_k_probs=0)
    assert cap.top_k_probs == 0


def test_capture_config_top_k_probs_negative() -> None:
    """Verify top_k_probs rejects negative values."""
    with pytest.raises(ValidationError):
        CaptureConfig(top_k_probs=-1)


def test_capture_config_layers_all() -> None:
    """Verify layers defaults to 'all' and accepts the literal."""
    cap = CaptureConfig()
    assert cap.layers == "all"


def test_capture_config_layers_list() -> None:
    """Verify layers accepts a list of integers."""
    cap = CaptureConfig(layers=[0, 1, 2])
    assert cap.layers == [0, 1, 2]


def test_capture_config_layers_empty_list() -> None:
    """Verify layers accepts an empty list."""
    cap = CaptureConfig(layers=[])
    assert cap.layers == []


def test_capture_config_defaults() -> None:
    """Verify CaptureConfig default values are set correctly."""
    cap = CaptureConfig()
    assert cap.hidden_states is True
    assert cap.attention_weights is True
    assert cap.qkv is True
    assert cap.mlp is True
    assert cap.logits is True
    assert cap.top_k_probs == 20
    assert cap.store_dtype == "float16"


# ---------------------------------------------------------------------------
# OutputConfig
# ---------------------------------------------------------------------------


def test_output_config_defaults() -> None:
    """Verify OutputConfig default values are set correctly."""
    out = OutputConfig()
    assert out.dir == "./runs"
    assert out.format == "json+npz"


def test_output_config_custom_dir() -> None:
    """Verify OutputConfig accepts a custom output directory."""
    out = OutputConfig(dir="/tmp/my_runs")
    assert out.dir == "/tmp/my_runs"


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


def _make_scenario(**kwargs) -> Scenario:
    """Helper to build a minimal valid Scenario, overridable via kwargs."""
    defaults = dict(
        name="test_scenario",
        model=ModelConfig(id="test_model"),
        prompt=[PromptConfig(messages=[Message(role="user", content="test")])],
    )
    defaults.update(kwargs)
    return Scenario(**defaults)


def test_scenario_valid() -> None:
    """Verify a minimal valid Scenario is created with correct field values."""
    scenario = _make_scenario()
    assert scenario.name == "test_scenario"
    assert scenario.model.id == "test_model"
    assert len(scenario.prompt) == 1


@pytest.mark.parametrize(
    "bad_name",
    [
        "test scenario",   # space
        "test/scenario",   # forward slash
        "test\\scenario",  # backslash
        "test\tscenario",  # tab
        "test\nscenario",  # newline
        "",                # empty string
    ],
)
def test_scenario_invalid_names(bad_name: str) -> None:
    """Verify that scenario names with whitespace, slashes, or empty strings are rejected."""
    with pytest.raises(ValidationError):
        _make_scenario(name=bad_name)


def test_scenario_prompt_dict_normalized_to_list() -> None:
    """Verify that a bare dict prompt is normalized to a single-element list."""
    scenario_dict = {
        "name": "test",
        "model": {"id": "test_model"},
        "prompt": {"messages": [{"role": "user", "content": "test"}]},
    }
    scenario = Scenario.model_validate(scenario_dict)
    assert isinstance(scenario.prompt, list)
    assert len(scenario.prompt) == 1


def test_scenario_prompt_list_preserved() -> None:
    """Verify that a list prompt is kept as-is (not double-wrapped)."""
    scenario = _make_scenario(
        prompt=[
            PromptConfig(messages=[Message(role="user", content="a")]),
            PromptConfig(messages=[Message(role="user", content="b")]),
        ]
    )
    assert len(scenario.prompt) == 2


def test_scenario_defaults() -> None:
    """Verify Scenario default sub-configs are instantiated correctly."""
    scenario = _make_scenario()
    assert isinstance(scenario.generation, GenerationConfig)
    assert isinstance(scenario.capture, CaptureConfig)
    assert isinstance(scenario.output, OutputConfig)
    assert scenario.reference_states == []


def test_scenario_empty_reference_states() -> None:
    """Verify reference_states defaults to an empty list."""
    scenario = _make_scenario()
    assert scenario.reference_states == []


def test_scenario_with_reference_states() -> None:
    """Verify reference_states accepts a list of ReferenceState objects."""
    ref = ReferenceState(label="ref1", messages=[Message(role="user", content="hi")])
    scenario = _make_scenario(reference_states=[ref])
    assert len(scenario.reference_states) == 1
    assert scenario.reference_states[0].label == "ref1"


# ---------------------------------------------------------------------------
# load_scenario
# ---------------------------------------------------------------------------

_VALID_YAML = """\
name: test_scenario
model:
  id: test_model
prompt:
  messages:
    - role: user
      content: test
generation: {}
capture: {}
output: {}
"""


def test_load_scenario_mocked() -> None:
    """Verify load_scenario parses YAML content into a valid Scenario.

    File I/O is mocked so no real filesystem access occurs.
    """
    with patch("builtins.open", mock_open(read_data=_VALID_YAML)):
        with patch("pathlib.Path.read_text", return_value=_VALID_YAML):
            scenario = load_scenario("fake_path.yaml")

    assert scenario.name == "test_scenario"
    assert scenario.model.id == "test_model"
    assert len(scenario.prompt) == 1
    assert scenario.prompt[0].messages[0].role == "user"


def test_load_scenario_invalid_yaml_raises() -> None:
    """Verify load_scenario raises an error when YAML produces an invalid Scenario."""
    bad_yaml = "name: 'bad name with spaces'\nmodel:\n  id: m\nprompt: []\n"
    with patch("pathlib.Path.read_text", return_value=bad_yaml):
        with pytest.raises(ValidationError):
            load_scenario("fake_path.yaml")


def test_load_scenario_missing_required_field() -> None:
    """Verify load_scenario raises ValidationError when a required field is absent."""
    yaml_no_model = "name: valid_name\nprompt:\n  messages:\n    - role: user\n      content: hi\n"
    with patch("pathlib.Path.read_text", return_value=yaml_no_model):
        with pytest.raises(ValidationError):
            load_scenario("fake_path.yaml")
