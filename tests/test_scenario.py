"""Unit tests for scenario module."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from activation_lab.scenario import (
    Message,
    PromptConfig,
    ReferenceState,
    ModelConfig,
    GenerationConfig,
    CaptureConfig,
    OutputConfig,
    Scenario,
    load_scenario,
)


def test_message_validation() -> None:
    """Test Message model validation."""
    # Valid message
    msg = Message(role="user", content="test")
    assert msg.role == "user"
    assert msg.content == "test"

    # Invalid role
    with pytest.raises(ValidationError):
        Message(role="invalid", content="test")


def test_prompt_config_validation() -> None:
    """Test PromptConfig model validation."""
    # Valid prompt config
    prompt = PromptConfig(messages=[Message(role="user", content="test")])
    assert len(prompt.messages) == 1


def test_reference_state_validation() -> None:
    """Test ReferenceState model validation."""
    # Valid reference state
    ref = ReferenceState(label="test_label", messages=[Message(role="user", content="test")])
    assert ref.label == "test_label"

    # Invalid label with whitespace
    with pytest.raises(ValidationError):
        ReferenceState(label="test label", messages=[])

    # Invalid label with slash
    with pytest.raises(ValidationError):
        ReferenceState(label="test/label", messages=[])

    # Invalid empty label
    with pytest.raises(ValidationError):
        ReferenceState(label="", messages=[])


def test_model_config_validation() -> None:
    """Test ModelConfig model validation."""
    # Valid model config
    model = ModelConfig(id="test_model")
    assert model.id == "test_model"

    # Test valid dtypes
    for dtype in ["float16", "bfloat16", "float32"]:
        model = ModelConfig(id="test", dtype=dtype)
        assert model.dtype == dtype

    # Invalid dtype
    with pytest.raises(ValidationError):
        ModelConfig(id="test", dtype="invalid_dtype")


def test_generation_config_validation() -> None:
    """Test GenerationConfig model validation."""
    # Valid generation config
    gen = GenerationConfig(max_new_tokens=5)
    assert gen.max_new_tokens == 5

    # Test min value constraint
    with pytest.raises(ValidationError):
        GenerationConfig(max_new_tokens=0)


def test_capture_config_validation() -> None:
    """Test CaptureConfig model validation."""
    # Valid capture config
    cap = CaptureConfig(top_k_probs=10)
    assert cap.top_k_probs == 10

    # Test valid store dtypes
    for dtype in ["float16", "bfloat16", "float32"]:
        cap = CaptureConfig(store_dtype=dtype)
        assert cap.store_dtype == dtype


def test_output_config_validation() -> None:
    """Test OutputConfig model validation."""
    # Valid output config
    out = OutputConfig()
    assert out.dir == "./runs"


def test_scenario_validation() -> None:
    """Test Scenario model validation."""
    # Valid scenario
    scenario = Scenario(
        name="test_scenario",
        model=ModelConfig(id="test_model"),
        prompt=[PromptConfig(messages=[Message(role="user", content="test")])]
    )
    assert scenario.name == "test_scenario"

    # Invalid name with whitespace
    with pytest.raises(ValidationError):
        Scenario(
            name="test scenario",
            model=ModelConfig(id="test_model"),
            prompt=[PromptConfig(messages=[Message(role="user", content="test")])]
        )

    # Invalid name with slash
    with pytest.raises(ValidationError):
        Scenario(
            name="test/scenario",
            model=ModelConfig(id="test_model"),
            prompt=[PromptConfig(messages=[Message(role="user", content="test")])]
        )


def test_scenario_normalization() -> None:
    """Test scenario prompt normalization."""
    # Test dict to list conversion
    scenario_dict = {
        "name": "test",
        "model": {"id": "test_model"},
        "prompt": {"messages": [{"role": "user", "content": "test"}]},
        "generation": {}
    }
    
    # This should work without errors
    scenario = Scenario.model_validate(scenario_dict)
    assert len(scenario.prompt) == 1


def test_load_scenario() -> None:
    """Test loading scenario from YAML file."""
    # Create a temporary YAML content for testing
    yaml_content = """
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
    
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    
    try:
        # This should work without errors
        scenario = load_scenario(temp_path)
        assert scenario.name == "test_scenario"
    finally:
        os.unlink(temp_path)