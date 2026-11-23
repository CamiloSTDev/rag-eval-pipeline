import pytest
from pydantic import SecretStr

from rag import bootstrap
from rag.application.judge import JudgeAnswer
from rag.settings import Settings


def settings(**values: object) -> Settings:
    # Los valores explícitos tienen prioridad sobre .env, así el test no depende de él
    base: dict[str, object] = {
        "deepseek_api_key": SecretStr("falsa"),
        "openai_api_key": SecretStr("falsa"),
        "llm_provider": "openai",
        "llm_model": "generador",
        "judge_provider": "deepseek",
        "judge_model": None,
    }
    return Settings(**(base | values))  # type: ignore[arg-type]


def test_without_judge_model_there_is_no_judge():
    assert bootstrap.build_judge(settings()) is None


def test_a_different_model_can_judge():
    judge = bootstrap.build_judge(settings(judge_model="juez"))

    assert isinstance(judge, JudgeAnswer)


def test_the_generator_cannot_judge_itself():
    with pytest.raises(bootstrap.ConfigurationError, match="distinto"):
        bootstrap.build_judge(
            settings(llm_provider="deepseek", llm_model="mismo", judge_model="mismo")
        )


def test_the_judge_needs_the_key_of_its_provider():
    with pytest.raises(bootstrap.ConfigurationError, match="DEEPSEEK_API_KEY"):
        bootstrap.build_judge(settings(judge_model="juez", deepseek_api_key=None))
