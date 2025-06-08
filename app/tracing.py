from dotenv import load_dotenv 
from traceloop.sdk import Traceloop
from opentelemetry import trace

if not(load_dotenv(verbose=True)):
    print("WARNING: .env not found, tracing is disabled")

Traceloop.init(
    app_name="PromptMind CLI",
    disable_batch=True,
)

tracer = trace.get_tracer(__name__)