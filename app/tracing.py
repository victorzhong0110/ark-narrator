"""OpenTelemetry 链路追踪（可选，env 开关）。

设 ARK_OTEL_ENDPOINT（如 http://jaeger:4318/v1/traces）且装了 otel 包时，自动给 FastAPI
打点并经 OTLP 上报；否则 no-op，不影响默认运行（otel 为可选依赖）。

安装：pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
              opentelemetry-instrumentation-fastapi
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def setup_tracing(app, service: str = "ark-narrator", endpoint: str | None = None) -> bool:
    """配置链路追踪。未配置端点或未装 otel → 返回 False（no-op）。"""
    endpoint = endpoint or os.getenv("ARK_OTEL_ENDPOINT", "")
    if not endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("未装 opentelemetry，跳过链路追踪（pip install opentelemetry-sdk …）")
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    logger.info("链路追踪已开启 → %s", endpoint)
    return True
