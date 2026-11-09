import asyncio
import hmac
import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.crew.guardrails import REQUEST_ID, validate_report
from src.crew.main import PROJECT_ROOT, _run_real_report
from src.crew.schemas import ExecutiveReport
from src.crew.tools import _get_numeric_top_ten, build_deterministic_report
from src.utils.db_gen import generate_synthetic_data


LOGGER = logging.getLogger(__name__)


def configure_safe_logging():
    previous_factory = logging.getLogRecordFactory()
    if getattr(previous_factory, '_api_safe', False):
        return

    def factory(*args, **kwargs):
        record = previous_factory(*args, **kwargs)
        request_id = REQUEST_ID.get()
        if request_id != 'cli' and record.name not in {
            __name__, 'src.crew.guardrails',
        }:
            record.msg = 'request_id=%s evento=dependencia status=%s'
            record.args = (request_id, record.levelname)
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return record

    factory._api_safe = True
    logging.setLogRecordFactory(factory)


class AnalysisInput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    top_n: int = Field(default=10, ge=1, le=20)
    use_llm: bool = False


@dataclass(frozen=True)
class Settings:
    api_access_key: str
    cors_origins: tuple[str, ...]
    allow_llm: bool
    db_path: Path
    max_calls_per_minute: int = 5
    timeout_seconds: float = 60

    @classmethod
    def from_environment(cls):
        load_dotenv(PROJECT_ROOT / '.env')
        try:
            origins = json.loads(os.getenv('CORS_ORIGINS', '[]'))
            allow_llm = os.getenv('ALLOW_LLM', 'false').strip().lower()
            database_url = os.getenv('DATABASE_URL', 'sqlite:///data/transacoes.db')
            limit = int(os.getenv('LLM_MAX_CALLS_PER_MINUTE', '5'))
            timeout = float(os.getenv('LLM_TIMEOUT_SECONDS', '60'))
            api_key = os.getenv('API_ACCESS_KEY', '')
            gemini_key = os.getenv('GEMINI_API_KEY', '')
            if api_key and gemini_key and api_key == gemini_key:
                raise ValueError
            if allow_llm not in {'true', 'false'}:
                raise ValueError
            if not isinstance(origins, list) or not all(
                isinstance(origin, str) and '*' not in origin
                and urlsplit(origin).scheme in {'http', 'https'}
                and urlsplit(origin).netloc
                and not urlsplit(origin).username
                and urlsplit(origin).path == ''
                and not urlsplit(origin).query and not urlsplit(origin).fragment
                for origin in origins
            ):
                raise ValueError
            if not database_url.startswith('sqlite:///') or limit < 1 or not 0 < timeout <= 300:
                raise ValueError
            db_path = Path(database_url.removeprefix('sqlite:///'))
            return cls(
                api_access_key=api_key,
                cors_origins=tuple(origins),
                allow_llm=allow_llm == 'true',
                db_path=(db_path if db_path.is_absolute() else PROJECT_ROOT / db_path).resolve(),
                max_calls_per_minute=limit,
                timeout_seconds=timeout,
            )
        except (ValueError, TypeError):
            raise RuntimeError('Configuração da API inválida.') from None


class CallLimitExceeded(RuntimeError):
    pass


class CallLimiter:
    def __init__(self, limit: int):
        self.limit = limit
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            while self.calls and self.calls[0] <= now - 60:
                self.calls.popleft()
            if len(self.calls) >= self.limit:
                raise CallLimitExceeded
            self.calls.append(now)


def deterministic_report(top_n: int, db_path: Path):
    ranking = _get_numeric_top_ten(db_path, top_n)
    return validate_report(build_deterministic_report(ranking), ranking, db_path, top_n)[0]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    configure_safe_logging()
    limiter = CallLimiter(settings.max_calls_per_minute)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            if not settings.db_path.exists():
                await run_in_threadpool(generate_synthetic_data, settings.db_path, seed=42)
        except Exception:
            LOGGER.error('request_id=%s evento=inicializacao_falhou', uuid4().hex)
            raise RuntimeError('Não foi possível inicializar a API.') from None
        yield

    app = FastAPI(lifespan=lifespan, debug=False)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=['GET', 'POST'],
        allow_headers=['Content-Type', 'X-API-Key'],
        expose_headers=['X-Request-ID'],
    )

    def error_response(request: Request, status: int, message: str):
        LOGGER.warning('request_id=%s status=%s', request.state.request_id, status)
        return JSONResponse(
            status_code=status,
            content={'detail': message, 'request_id': request.state.request_id},
            headers={'X-Request-ID': request.state.request_id},
        )

    @app.middleware('http')
    async def request_boundary(request: Request, call_next):
        request.state.request_id = uuid4().hex
        origin = request.headers.get('origin')
        if origin is not None and origin not in settings.cors_origins:
            return error_response(request, 403, 'Origem não permitida.')
        token = REQUEST_ID.set(request.state.request_id)
        try:
            response = await call_next(request)
        except Exception:
            return error_response(request, 500, 'Não foi possível processar a solicitação.')
        finally:
            REQUEST_ID.reset(token)
        response.headers['X-Request-ID'] = request.state.request_id
        LOGGER.info('request_id=%s status=%s', request.state.request_id, response.status_code)
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, _error: RequestValidationError):
        return error_response(request, 422, 'Entrada inválida.')

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException):
        messages = {
            401: 'Não autorizado.',
            403: 'Solicitação não permitida.',
            404: 'Recurso não encontrado.',
            405: 'Método não permitido.',
            429: 'Limite de chamadas atingido.',
            503: 'Serviço indisponível.',
            504: 'Tempo limite excedido.',
        }
        return error_response(request, error.status_code, messages.get(
            error.status_code, 'Não foi possível processar a solicitação.',
        ))

    def authenticate(request: Request):
        provided = request.headers.get('x-api-key', '')
        if not settings.api_access_key.strip():
            raise HTTPException(status_code=503)
        if not provided or not hmac.compare_digest(
            provided.encode('utf-8'), settings.api_access_key.encode('utf-8'),
        ):
            raise HTTPException(status_code=401)

    @app.get('/health')
    async def health():
        return {'status': 'ok'}

    @app.post('/analisar', response_model=ExecutiveReport, dependencies=[Depends(authenticate)])
    async def analyze(payload: AnalysisInput):
        if not payload.use_llm:
            return await run_in_threadpool(deterministic_report, payload.top_n, settings.db_path)
        if not settings.allow_llm:
            raise HTTPException(status_code=403)

        deadline = time.monotonic() + settings.timeout_seconds
        cancelled = threading.Event()
        rate_limited = threading.Event()
        timed_out = threading.Event()

        def guard():
            if cancelled.is_set() or time.monotonic() >= deadline:
                timed_out.set()
                raise TimeoutError
            try:
                limiter.acquire()
            except CallLimitExceeded:
                rate_limited.set()
                raise

        try:
            report, _ = await asyncio.wait_for(
                run_in_threadpool(
                    _run_real_report, top_n=payload.top_n, db_path=settings.db_path,
                    call_guard=guard, timeout=settings.timeout_seconds,
                ),
                timeout=settings.timeout_seconds,
            )
            if timed_out.is_set():
                raise HTTPException(status_code=504)
            if rate_limited.is_set():
                raise HTTPException(status_code=429)
            return report
        except CallLimitExceeded:
            raise HTTPException(status_code=429) from None
        except TimeoutError:
            raise HTTPException(status_code=504) from None
        except Exception:
            if timed_out.is_set():
                raise HTTPException(status_code=504) from None
            if rate_limited.is_set():
                raise HTTPException(status_code=429) from None
            raise
        finally:
            cancelled.set()

    return app


app = create_app()
