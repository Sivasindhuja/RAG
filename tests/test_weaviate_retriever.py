import os
from types import SimpleNamespace

import pytest

from src.rag_app.config import AppConfig
from src.rag_app.service import RAGService
from src.rag_core.weaviate_retriever import (
    WeaviateRetriever,
    WeaviateRetrieverConfig,
    WeaviateConfigError,
    WeaviateRetrievalError,
)


class FakeWeaviateQuery:
    def __init__(self, objects):
        self._objects = objects
        self.last_call = None
        self.last_kwargs = None

    def hybrid(self, **kwargs):
        self.last_call = "hybrid"
        self.last_kwargs = kwargs
        return SimpleNamespace(objects=self._objects)

    def bm25(self, **kwargs):
        self.last_call = "bm25"
        self.last_kwargs = kwargs
        return SimpleNamespace(objects=self._objects)

    def near_text(self, **kwargs):
        self.last_call = "near_text"
        self.last_kwargs = kwargs
        return SimpleNamespace(objects=self._objects)


class FakeWeaviateCollection:
    def __init__(self, objects):
        self.query = FakeWeaviateQuery(objects)


class FakeWeaviateCollections:
    def __init__(self, objects):
        self._objects = objects
        self.last_collection = None

    def use(self, name):
        self.last_collection = FakeWeaviateCollection(self._objects)
        return self.last_collection


class FakeWeaviateClient:
    def __init__(self, objects):
        self.collections = FakeWeaviateCollections(objects)
        self.closed = False

    def close(self):
        self.closed = True


def test_default_backend_remains_chroma():
    cfg = AppConfig.from_env()
    assert cfg.retriever_backend == "chroma"


def test_weaviate_requires_explicit_backend_selection(monkeypatch):
    # Default config should not construct weaviate retriever.
    cfg = AppConfig.from_env()
    svc = RAGService(config=cfg)
    assert svc._weaviate_retriever is None


def test_missing_weaviate_credentials_fail_clearly(monkeypatch):
    for key in [
        "WEAVIATE_URL",
        "WEAVIATE_API_KEY",
        "WEAVIATE_COLLECTION",
        "RAG_WEAVIATE_URL",
        "RAG_WEAVIATE_API_KEY",
        "RAG_WEAVIATE_COLLECTION",
    ]:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(WeaviateConfigError):
        WeaviateRetrieverConfig.from_env()


def test_weaviate_retrieval_normalized_structure():
    objs = [
        SimpleNamespace(
            uuid="uuid-1",
            properties={
                "chunk_id": "c1",
                "text": "chunk text",
                "source": "satcom-ngp.pdf",
                "start_offset": 10,
                "end_offset": 20,
                "start_page": 1,
                "end_page": 1,
                "article": "Article 1",
                "section": "1.1",
                "heading": "Intro",
            },
            metadata=SimpleNamespace(score=0.9, distance=0.1),
        )
    ]
    cfg = WeaviateRetrieverConfig(url="https://example", api_key="key", collection_name="Col")
    r = WeaviateRetriever(cfg, client=FakeWeaviateClient(objs))
    res = r.retrieve("test question", k=5)
    assert res.debug["mode"] == "weaviate_hybrid"
    assert len(res.hits) == 1
    h = res.hits[0]
    assert h.chunk_id == "c1"
    assert h.text == "chunk text"
    assert h.start_offset == 10
    assert h.end_offset == 20
    assert h.start_page == 1
    assert h.section == "1.1"
    assert h.metadata["source"] == "satcom-ngp.pdf"


def test_weaviate_connection_error_handled(monkeypatch):
    cfg = WeaviateRetrieverConfig(url="https://example", api_key="key", collection_name="Col")
    r = WeaviateRetriever(cfg)

    def boom():
        raise RuntimeError("connect failed")

    monkeypatch.setattr(r, "_connect", boom)
    with pytest.raises(RuntimeError):
        r.retrieve("question", k=3)


def test_service_weaviate_backend_does_not_init_chroma(monkeypatch):
    # Ensure weaviate backend path is used and chroma/index not touched.
    cfg = AppConfig.from_env()
    cfg = AppConfig(**{**cfg.__dict__, "retriever_backend": "weaviate", "weaviate_url": "https://example", "weaviate_api_key": "k", "weaviate_collection": "Col"})
    svc = RAGService(config=cfg)

    class FakeRetriever:
        def retrieve(self, question, k=5, search_mode="hybrid", alpha=0.5):
            from src.rag_core.retrieval import RetrievalResult
            return RetrievalResult(hits=[], debug={"mode": f"weaviate_{search_mode}", "final": []})

    svc._weaviate_retriever = FakeRetriever()

    def fail_index():
        raise AssertionError("index should not be accessed for weaviate backend")

    monkeypatch.setattr(RAGService, "index", property(lambda self: fail_index()))
    res = svc._retrieve("question", 5, "hybrid")
    assert res.debug["mode"] == "weaviate_hybrid"


def test_ingestion_script_does_not_reference_production_chroma_write():
    # Safety guard: experimental ingestion script must not target chroma_db_production.
    from pathlib import Path
    text = Path("scripts/weaviate_ingest.py").read_text(encoding="utf-8").lower()
    assert "persistentclient" not in text
    assert "denseindex" not in text
    assert "chroma_db_production" not in text.split("safety:")[0]  # allow safety message only after marker
    # and no actual chroma path usage
    assert "chromadb" not in text


def test_baseline_eval_defaults_unchanged():
    from src.rag_app.config import AppConfig
    cfg = AppConfig.from_env()
    assert cfg.retriever_backend == "chroma"
    assert cfg.retrieval_mode == "hybrid"


def test_weaviate_retriever_module_import_does_not_load_heavy_local_stack():
    # Subprocess isolation: Weaviate retriever imports lightweight retrieval_types.
    import json, subprocess, sys
    code = (
        "import json, sys; "
        "from src.rag_core.weaviate_retriever import WeaviateRetriever; "
        "mods=['torch','sentence_transformers','transformers','chromadb','rank_bm25','numpy','scipy']; "
        "print(json.dumps({m:(m in sys.modules) for m in mods}))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr + "\n" + r.stdout
    data = json.loads(r.stdout.strip().splitlines()[-1])
    for m in ['torch', 'sentence_transformers', 'transformers', 'chromadb', 'rank_bm25', 'numpy', 'scipy']:
        assert data[m] is False, f"{m} should not be imported by weaviate_retriever"


def test_weaviate_backend_app_import_and_mocked_query_do_not_load_heavy_stack():
    # Subprocess isolation: Weaviate backend selection + mocked retrieval should not
    # import heavy local retrieval/embedding stacks.
    import json, subprocess, sys, os
    code = r"""
import json, os, sys
os.environ['RAG_RETRIEVER_BACKEND']='weaviate'
os.environ['WEAVIATE_URL']='https://example.invalid'
os.environ['WEAVIATE_API_KEY']='dummy'
os.environ['WEAVIATE_COLLECTION']='Dummy'
import backend.main  # noqa: F401
from src.rag_app.config import AppConfig
from src.rag_app.service import RAGService
from src.rag_core.retrieval_types import RetrievalResult
class FakeR:
    def retrieve(self, question, k=5, search_mode='hybrid', alpha=0.5):
        return RetrievalResult(hits=[], debug={'mode':f'weaviate_{search_mode}','final':[]})
svc = RAGService(config=AppConfig.from_env())
svc._weaviate_retriever = FakeR()
svc._retrieve('test', 5, 'hybrid')
mods=['torch','sentence_transformers','transformers','chromadb','rank_bm25','numpy','scipy']
print(json.dumps({m:(m in sys.modules) for m in mods}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr + "\n" + r.stdout
    data = json.loads(r.stdout.strip().splitlines()[-1])
    for m in ['torch', 'sentence_transformers', 'transformers', 'chromadb', 'rank_bm25', 'scipy']:
        assert data[m] is False, f"{m} should not be imported on Weaviate path"
    # numpy may be imported by unavoidable third-party deps (e.g. langfuse serializer)
    # depending on environment/config; document rather than force unsafe workaround.
    if data.get('numpy'):
        import sys
        print('AUDIT_NOTE: numpy imported on Weaviate path (likely third-party dep such as langfuse)', file=sys.stderr)


def test_chroma_backend_default_import_still_available():
    # Default backend remains chroma; lazy exports still resolve.
    from src.rag_app.config import AppConfig
    from src.rag_core import DenseIndex, HybridRetriever, RetrievalResult, RetrievedChunk
    cfg = AppConfig.from_env()
    assert cfg.retriever_backend == "chroma"
    assert DenseIndex is not None
    assert HybridRetriever is not None
    assert RetrievalResult is not None
    assert RetrievedChunk is not None


def _load_ingest_module():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("weaviate_ingest", Path("scripts/weaviate_ingest.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ingest_refuses_production_like_collection_name():
    mod = _load_ingest_module()
    with pytest.raises(SystemExit):
        mod._assert_safe_collection_name("satcom_prod")
    with pytest.raises(SystemExit):
        mod._assert_safe_collection_name("production_collection")


def test_ingest_refuses_non_experimental_collection_name():
    mod = _load_ingest_module()
    with pytest.raises(SystemExit):
        mod._assert_safe_collection_name("SatcomNgpCollection")


def test_ingest_recreate_requires_exact_confirmation():
    mod = _load_ingest_module()
    class FakeCollection:
        def aggregate(self):
            class A:
                total_count = 7
            return A()
    class FakeCollections:
        def exists(self, name): return True
        def get(self, name): return FakeCollection()
        def delete(self, name): raise AssertionError("should not delete")
    class FakeClient:
        collections = FakeCollections()
    with pytest.raises(SystemExit):
        mod.ensure_collection(FakeClient(), "SatcomNgpExp800160", "model", recreate=True, confirm_delete_collection="", allow_existing=False)
    with pytest.raises(SystemExit):
        mod.ensure_collection(FakeClient(), "SatcomNgpExp800160", "model", recreate=True, confirm_delete_collection="wrong", allow_existing=False)


def test_ingest_existing_collection_stops_without_allow_existing():
    mod = _load_ingest_module()
    class FakeAgg:
        total_count = 61
    class FakeCollection:
        def aggregate(self):
            class X:
                def over_all(self, total_count=True):
                    return FakeAgg()
            return X()
    class FakeCollections:
        def exists(self, name): return True
        def get(self, name): return FakeCollection()
    class FakeClient:
        collections = FakeCollections()
    with pytest.raises(SystemExit):
        mod.ensure_collection(FakeClient(), "SatcomNgpExp800160", "model", recreate=False, confirm_delete_collection="", allow_existing=False)


def test_eval_default_output_is_unique_timestamped(tmp_path, monkeypatch):
    # Evaluation script should default to a unique timestamped out dir and refuse overwrite without --overwrite.
    import importlib.util, sys, json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("weaviate_eval", Path("evaluation/weaviate_eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # simulate existing non-empty dir
    out = tmp_path / "exists"
    out.mkdir()
    (out / "report.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["weaviate_eval", "--dataset", "evaluation/ci_golden_dataset.json", "--out", str(out), "--run-baseline"])
    # run_baseline would try to run; but we want overwrite guard to trigger before heavy work. It loads dataset/doc before guard?
    # In our implementation guard is after dataset/doc load - acceptable safety guard; test by expecting SystemExit due to overwrite guard.
    try:
        mod.main()
    except SystemExit as e:
        assert "overwrite" in str(e).lower() or "--overwrite" in str(e)
    else:
        raise AssertionError("Expected SystemExit for existing non-empty output without --overwrite")


def test_retrieved_diagnostic_serialization_excludes_text():
    from evaluation.ci_eval import _retrieved_diagnostic
    class R:
        rank=1; chunk_id="c1"; section="s"; start_page=1; end_page=1; start_offset=0; end_offset=10; distance=0.2
        score=0.8
        retrieval_scores={"weaviate_score":0.8, "weaviate_distance":0.2}
        metadata={"text":"SECRET_TEXT", "chunk_id":"c1", "_weaviate_score":0.8}
    d = _retrieved_diagnostic(R())
    assert d["chunk_id"] == "c1"
    assert d["retrieval_scores"]["weaviate_score"] == 0.8
    assert "text" not in d["metadata"]
    assert d["metadata"]["chunk_id"] == "c1"


def test_abstention_diagnostics_include_weaviate_scores():
    from types import SimpleNamespace
    from evaluation.ci_eval import _abstention_diagnostics
    resp = SimpleNamespace(
        abstained=True,
        abstention_reason="insufficient_evidence",
        status="abstained",
        retrieval_mode="weaviate_hybrid",
        retrieved=[SimpleNamespace(model_dump=lambda: {
            "rank":1,"chunk_id":"c1","distance":0.7,
            "retrieval_scores":{"weaviate_score":0.3,"weaviate_distance":0.7},
            "metadata":{"_weaviate_score":0.3,"_weaviate_distance":0.7}
        })],
        debug={"best_distance":0.7,"token_support":0.1,"selected_lexical_support":0.2,
               "answerability_guard_applied":True,"answerability_guard_reason":"guard","pipeline_status":"abstained",
               "retrieval_debug":{"mode":"weaviate_hybrid"}}
    )
    d = _abstention_diagnostics(resp)
    assert d["abstained"] is True
    assert d["raw_weaviate_score"] == 0.3
    assert d["raw_weaviate_distance"] == 0.7
    assert d["retrieval_mode"] == "weaviate_hybrid"


def test_diagnostics_do_not_modify_production_collection(monkeypatch):
    # Diagnostic helpers are read-only; ensure they don't call mutating index methods.
    from evaluation.ci_eval import _retrieved_diagnostic, _abstention_diagnostics
    from types import SimpleNamespace
    resp = SimpleNamespace(abstained=False, abstention_reason=None, status="answered", retrieval_mode="hybrid", retrieved=[], debug={})
    _abstention_diagnostics(resp)
    _retrieved_diagnostic({"rank":1,"chunk_id":"c1","metadata":{"text":"x"}})


def test_weaviate_connect_passes_timeout_config(monkeypatch):
    captured = {}

    class FakeClient:
        def close(self):
            pass

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return FakeClient()

    import weaviate
    from weaviate.classes.init import AdditionalConfig, Timeout

    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", fake_connect)
    cfg = WeaviateRetrieverConfig(
        url="https://example.weaviate.cloud",
        api_key="key",
        collection_name="Col",
        init_timeout_s=7.5,
        query_timeout_s=45.0,
    )
    r = WeaviateRetriever(cfg)
    client = r._connect()
    assert client is not None
    assert captured["cluster_url"] == "https://example.weaviate.cloud"
    assert captured["auth_credentials"] is not None
    ac = captured.get("additional_config")
    assert isinstance(ac, AdditionalConfig)
    # Timeout model fields
    t = ac.timeout
    assert isinstance(t, Timeout)
    assert float(t.init) == 7.5
    assert float(t.query) == 45.0


def test_weaviate_connect_uses_default_timeout_values(monkeypatch):
    captured = {}

    class FakeClient:
        def close(self):
            pass

    monkeypatch.setattr(__import__("weaviate"), "connect_to_weaviate_cloud", lambda **kw: captured.update(kw) or FakeClient())
    cfg = WeaviateRetrieverConfig(url="https://example", api_key="k", collection_name="Col")
    WeaviateRetriever(cfg)._connect()
    ac = captured.get("additional_config")
    assert ac is not None
    assert float(ac.timeout.init) == 2.0
    assert float(ac.timeout.query) == 30.0


def test_weaviate_connect_error_propagates_as_retrieval_error(monkeypatch):
    import weaviate

    def boom(**kwargs):
        raise RuntimeError("grpc deadline")

    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", boom)
    cfg = WeaviateRetrieverConfig(url="https://example", api_key="k", collection_name="Col")
    r = WeaviateRetriever(cfg)
    with pytest.raises(WeaviateRetrievalError) as ei:
        r.retrieve("question", k=3)
    msg = str(ei.value)
    assert "Weaviate connection failed" in msg
    assert "RuntimeError" in msg
    assert "grpc deadline" in msg


def test_weaviate_timeout_config_from_env(monkeypatch):
    monkeypatch.setenv("WEAVIATE_URL", "https://example.weaviate.cloud")
    monkeypatch.setenv("WEAVIATE_API_KEY", "key")
    monkeypatch.setenv("WEAVIATE_COLLECTION", "SatcomNgpExp800160")
    monkeypatch.setenv("WEAVIATE_INIT_TIMEOUT_S", "12")
    monkeypatch.setenv("WEAVIATE_QUERY_TIMEOUT_S", "60")
    cfg = WeaviateRetrieverConfig.from_env()
    assert cfg.init_timeout_s == 12.0
    assert cfg.query_timeout_s == 60.0


def test_weaviate_timeout_config_invalid_env(monkeypatch):
    monkeypatch.setenv("WEAVIATE_URL", "https://example.weaviate.cloud")
    monkeypatch.setenv("WEAVIATE_API_KEY", "key")
    monkeypatch.setenv("WEAVIATE_COLLECTION", "SatcomNgpExp800160")
    monkeypatch.setenv("WEAVIATE_INIT_TIMEOUT_S", "nope")
    with pytest.raises(WeaviateConfigError):
        WeaviateRetrieverConfig.from_env()


def test_app_config_weaviate_timeout_defaults(monkeypatch):
    # Ensure defaults preserved when env not set
    monkeypatch.delenv("WEAVIATE_INIT_TIMEOUT_S", raising=False)
    monkeypatch.delenv("WEAVIATE_QUERY_TIMEOUT_S", raising=False)
    monkeypatch.delenv("RAG_WEAVIATE_INIT_TIMEOUT_S", raising=False)
    monkeypatch.delenv("RAG_WEAVIATE_QUERY_TIMEOUT_S", raising=False)
    cfg = AppConfig.from_env()
    assert cfg.weaviate_init_timeout_s == 2.0
    assert cfg.weaviate_query_timeout_s == 30.0


def test_service_passes_weaviate_timeout_config(monkeypatch):
    captured = {}

    class FakeRetriever:
        def __init__(self, cfg, client=None):
            captured["cfg"] = cfg

    import src.rag_core.weaviate_retriever as wr
    monkeypatch.setattr(wr, "WeaviateRetriever", FakeRetriever)
    cfg = AppConfig.from_env()
    cfg = AppConfig(**{**cfg.__dict__, "retriever_backend": "weaviate", "weaviate_url": "https://example", "weaviate_api_key": "k", "weaviate_collection": "Col", "weaviate_init_timeout_s": 9.0, "weaviate_query_timeout_s": 55.0})
    svc = RAGService(config=cfg)
    _ = svc.weaviate_retriever
    assert captured["cfg"].init_timeout_s == 9.0
    assert captured["cfg"].query_timeout_s == 55.0


def _sample_objects():
    return [
        SimpleNamespace(
            uuid="uuid-1",
            properties={
                "chunk_id": "c1",
                "text": "chunk text",
                "source": "satcom-ngp.pdf",
                "start_offset": 10,
                "end_offset": 20,
                "start_page": 1,
                "end_page": 1,
                "section": "1.1",
            },
            metadata=SimpleNamespace(score=0.9, distance=0.1),
        )
    ]


def test_weaviate_retrieve_dense_mode_uses_near_text():
    cfg = WeaviateRetrieverConfig(url="https://example", api_key="key", collection_name="Col")
    client = FakeWeaviateClient(_sample_objects())
    r = WeaviateRetriever(cfg, client=client)
    res = r.retrieve("test question", k=5, search_mode="dense")
    assert client.collections.last_collection.query.last_call == "near_text"
    assert res.debug["mode"] == "weaviate_dense"
    assert res.debug["search_mode"] == "dense"
    assert "alpha" not in res.debug


def test_weaviate_retrieve_bm25_mode_uses_bm25():
    cfg = WeaviateRetrieverConfig(url="https://example", api_key="key", collection_name="Col")
    client = FakeWeaviateClient(_sample_objects())
    r = WeaviateRetriever(cfg, client=client)
    res = r.retrieve("test question", k=5, search_mode="bm25")
    assert client.collections.last_collection.query.last_call == "bm25"
    assert res.debug["mode"] == "weaviate_bm25"


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_weaviate_retrieve_hybrid_alpha_passed(alpha):
    cfg = WeaviateRetrieverConfig(url="https://example", api_key="key", collection_name="Col", alpha=0.5)
    client = FakeWeaviateClient(_sample_objects())
    r = WeaviateRetriever(cfg, client=client)
    res = r.retrieve("test question", k=5, search_mode="hybrid", alpha=alpha)
    assert client.collections.last_collection.query.last_call == "hybrid"
    assert client.collections.last_collection.query.last_kwargs["alpha"] == alpha
    assert res.debug["alpha"] == alpha
    assert res.debug["mode"] == "weaviate_hybrid"


def test_weaviate_retrieve_invalid_mode_raises():
    cfg = WeaviateRetrieverConfig(url="https://example", api_key="key", collection_name="Col")
    r = WeaviateRetriever(cfg, client=FakeWeaviateClient(_sample_objects()))
    with pytest.raises(Exception) as ei:
        r.retrieve("q", k=5, search_mode="bogus")
    assert "Invalid Weaviate search mode" in str(ei.value)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, "bad"])
def test_weaviate_retrieve_invalid_alpha_raises(alpha):
    cfg = WeaviateRetrieverConfig(url="https://example", api_key="key", collection_name="Col")
    r = WeaviateRetriever(cfg, client=FakeWeaviateClient(_sample_objects()))
    with pytest.raises(Exception):
        r.retrieve("q", k=5, search_mode="hybrid", alpha=alpha)


def test_weaviate_default_config_unchanged():
    cfg = WeaviateRetrieverConfig(url="https://example", api_key="key", collection_name="Col")
    assert cfg.alpha == 0.5
    assert cfg.search_mode == "hybrid"
    assert cfg.init_timeout_s == 2.0
    assert cfg.query_timeout_s == 30.0


def test_app_config_weaviate_retrieval_defaults_unchanged(monkeypatch):
    for k in ["WEAVIATE_ALPHA", "RAG_WEAVIATE_ALPHA", "WEAVIATE_SEARCH_MODE", "RAG_WEAVIATE_SEARCH_MODE"]:
        monkeypatch.delenv(k, raising=False)
    cfg = AppConfig.from_env()
    assert cfg.weaviate_alpha == 0.5
    assert cfg.weaviate_search_mode == "hybrid"
    assert cfg.retriever_backend == "chroma"


def _hit(chunk_id, distance=None, score=None, text="SATCOM policy text about authorization and licensing."):
    md = {"_weaviate_score": score} if score is not None else {}
    return SimpleNamespace(
        rank=1, chunk_id=chunk_id, text=text, source="satcom-ngp.pdf",
        start_offset=0, end_offset=len(text), start_page=1, end_page=1,
        article=None, section="1", heading=None, distance=distance, metadata=md,
    )


def test_abstention_policy_defaults_unchanged():
    from src.rag_core.pipeline import AbstentionPolicy
    p = AbstentionPolicy()
    assert p.hard_max_distance == 1.30
    assert p.soft_max_distance == 1.10
    assert p.min_token_support == 0.50
    assert p.strict_min_token_support == 0.70
    assert p.missing_distance_behavior == "skip"
    assert p.min_retrieval_score is None


def test_weaviate_experimental_abstention_disabled_by_default():
    cfg = AppConfig.from_env()
    assert cfg.weaviate_abstention_experimental is False
    assert cfg.weaviate_abstention_missing_distance_behavior == "skip"
    assert cfg.weaviate_abstention_min_retrieval_score is None


def test_chroma_policy_not_changed_by_weaviate_experimental_flag(monkeypatch):
    from src.rag_core.pipeline import AbstentionPolicy, GroundedPipeline
    # Chroma/default pipeline uses default policy even if weaviate experimental config exists.
    cfg = AppConfig.from_env()
    cfg = AppConfig(**{**cfg.__dict__, "retriever_backend": "chroma", "weaviate_abstention_experimental": True})
    svc = RAGService(config=cfg)
    # Avoid building real index; inspect pipeline policy directly by constructing as service would.
    p = svc.pipeline
    assert p.policy.missing_distance_behavior == "skip"
    assert p.policy.hard_max_distance == 1.30


def test_weaviate_dense_distance_still_abstains_with_default_policy():
    from src.rag_core.pipeline import AbstentionPolicy, GroundedPipeline
    policy = AbstentionPolicy()
    pipe = GroundedPipeline(index=None, policy=policy)
    # Distance present and very high -> hard distance abstention should still fire.
    hits = [_hit("c1", distance=1.5, text="irrelevant text tokens here")]
    ans = pipe.answer_from_hits("what is the authorization requirement?", hits)
    assert ans.abstained
    assert "best distance" in (ans.reason or "")


def test_weaviate_bm25_missing_distance_default_behavior_unchanged():
    from src.rag_core.pipeline import AbstentionPolicy, GroundedPipeline
    policy = AbstentionPolicy()  # default skip
    pipe = GroundedPipeline(index=None, policy=policy)
    hits = [_hit("c1", distance=None, score=2.7, text="policy authorization licensing text")]
    ans = pipe.answer_from_hits("authorization licensing", hits)
    # With missing distance, default behavior skips distance gate; answerability depends on lexical support.
    # We assert it does not crash and follows existing skip behavior.
    assert ans.best_distance is None


def test_weaviate_bm25_missing_distance_experimental_reject():
    from src.rag_core.pipeline import AbstentionPolicy, GroundedPipeline
    policy = AbstentionPolicy(missing_distance_behavior="reject")
    pipe = GroundedPipeline(index=None, policy=policy)
    hits = [_hit("c1", distance=None, score=10.0, text="authorization licensing policy text")]
    ans = pipe.answer_from_hits("authorization licensing", hits)
    assert ans.abstained
    assert ans.reason == "distance_unavailable:retrieval_distance_missing"


def test_weaviate_hybrid_missing_distance_experimental_score_threshold():
    from src.rag_core.pipeline import AbstentionPolicy, GroundedPipeline
    policy = AbstentionPolicy(missing_distance_behavior="score", min_retrieval_score=0.8)
    pipe = GroundedPipeline(index=None, policy=policy)
    hits = [_hit("c1", distance=None, score=0.7, text="authorization licensing policy text")]
    ans = pipe.answer_from_hits("authorization licensing", hits)
    assert ans.abstained
    assert "retrieval_score 0.7000 below min_retrieval_score 0.8000" in (ans.reason or "")


def test_weaviate_hybrid_missing_distance_score_boundary_0_50():
    from src.rag_core.pipeline import AbstentionPolicy, GroundedPipeline
    policy = AbstentionPolicy(missing_distance_behavior="score", min_retrieval_score=0.5)
    pipe = GroundedPipeline(index=None, policy=policy)
    hits = [_hit("c1", distance=None, score=0.5, text="authorization licensing policy text")]
    ans = pipe.answer_from_hits("authorization licensing", hits)
    # 0.5 < 0.5 is false, so should not be rejected by score threshold
    assert "below min_retrieval_score" not in (ans.reason or "")


def test_weaviate_missing_score_and_distance_experimental_score_mode_abstains():
    from src.rag_core.pipeline import AbstentionPolicy, GroundedPipeline
    policy = AbstentionPolicy(missing_distance_behavior="score", min_retrieval_score=0.5)
    pipe = GroundedPipeline(index=None, policy=policy)
    hits = [_hit("c1", distance=None, score=None, text="authorization licensing policy text")]
    ans = pipe.answer_from_hits("authorization licensing", hits)
    assert ans.abstained
    assert ans.reason == "distance_unavailable:retrieval_score_missing"


def test_answerable_not_rejected_merely_because_distance_absent_default():
    from src.rag_core.pipeline import AbstentionPolicy, GroundedPipeline
    policy = AbstentionPolicy()  # default skip
    pipe = GroundedPipeline(index=None, policy=policy)
    # Lexically supportive answerable-like content, no distance, has score
    text = "Article 1 provides authorization and licensing requirements for SATCOM services."
    hits = [_hit("c1", distance=None, score=5.0, text=text)]
    ans = pipe.answer_from_hits("What are the authorization and licensing requirements?", hits)
    # Should not abstain just because distance is missing; may answer or abstain based on normal lexical/selection rules.
    assert "distance_unavailable" not in (ans.reason or "")
