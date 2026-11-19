from src.rag_app.langfuse_tracing import LangfuseTracer


def test_langfuse_tracer_noop_without_credentials(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    tracer = LangfuseTracer(enabled=True)
    assert tracer.available is False
    trace = tracer.start_trace("t", input={"q": "x"})
    with tracer.span(trace, "s", input={"a": 1}) as span:
        span.update(output={"b": 2})
    tracer.flush()  # should not raise


def test_langfuse_tracer_disabled(monkeypatch):
    tracer = LangfuseTracer(enabled=False)
    assert tracer.available is False
    assert tracer.status()["enabled"] is False
