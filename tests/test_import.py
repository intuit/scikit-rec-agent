def test_package_imports():
    import scikit_rec_agent

    expected = {
        "Agent",
        "AgentEvent",
        "BaseLLM",
        "LLMResponse",
        "LLMStreamEvent",
        "ToolCall",
        "DEFAULT_SYSTEM_PROMPT",
        "DEFAULT_TOOLS",
        "DatasetBundle",
        "ModelHandle",
        "Session",
        "Tool",
        "get_default_tools",
        "ok",
        "err",
    }
    assert expected.issubset(set(scikit_rec_agent.__all__))
