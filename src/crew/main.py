import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from src.crew.guardrails import generate_validated_report, validate_report
from src.crew.schemas import ExecutiveReport

from src.crew.tools import (
    _get_numeric_top_ten,
    build_deterministic_report,
    create_numeric_ranking_tool,
    verify_report_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENTS_CONFIG = PROJECT_ROOT / 'config' / 'agents.yaml'
TASKS_CONFIG = PROJECT_ROOT / 'config' / 'tasks.yaml'
MCP_SERVER = PROJECT_ROOT / 'mcp_servers' / 'sqlite_server.py'


def _read_configuration(path):
    with path.open(encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f'Configuração YAML inválida: {path.name}.')
    return config


def _prepare_model():
    load_dotenv(PROJECT_ROOT / '.env')
    model_name = os.getenv('LLM_MODEL', '').strip()
    if not model_name:
        raise RuntimeError(
            'LLM_MODEL não está definido no .env; configure-o para a execução real '
            'ou use --dry-run.'
        )
    if not os.getenv('GEMINI_API_KEY', '').strip():
        raise RuntimeError(
            'GEMINI_API_KEY não está configurada no .env; configure-a para a '
            'execução real ou use --dry-run.'
        )
    if not model_name.startswith('gemini/'):
        raise RuntimeError('LLM_MODEL deve usar o prefixo gemini/ nesta integração; use --dry-run.')
    from google.genai.types import HttpOptions, HttpRetryOptions
    from src.crew.llm import ReviewGeminiCompletion

    return ReviewGeminiCompletion(
        model=model_name.removeprefix('gemini/'),
        temperature=0.1,
        max_output_tokens=2048,
        timeout=60,
        use_vertexai=False,
        client_params={
            'http_options': HttpOptions(
                timeout=60_000,
                retry_options=HttpRetryOptions(attempts=1),
            ),
        },
    )


def _validate_llm_report(candidate, ranking):
    return validate_report(candidate, ranking)[0]


def _run_real_report():
    llm = _prepare_model()
    from crewai import Agent, Crew, Process, Task
    from crewai_tools.adapters.mcp_adapter import MCPServerAdapter
    from mcp import StdioServerParameters

    ranking_tool = create_numeric_ranking_tool()
    numeric_tool_result = ranking_tool.run()
    ranking = json.loads(numeric_tool_result)
    if len(ranking) != 10:
        raise ValueError('A ferramenta numérica não retornou exatamente dez casos.')

    agents_config = _read_configuration(AGENTS_CONFIG)
    tasks_config = _read_configuration(TASKS_CONFIG)
    mcp_parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER)],
        cwd=str(PROJECT_ROOT),
    )
    with MCPServerAdapter(mcp_parameters) as mcp_tools:
        analyst = Agent(
            config=agents_config['analista_sql'],
            llm=llm,
            tools=list(mcp_tools),
            verbose=False,
        )
        scientist = Agent(
            config=agents_config['cientista_dados'],
            llm=llm,
            tools=[ranking_tool],
            verbose=False,
        )
        writer = Agent(
            config=agents_config['redator_executivo'],
            llm=llm,
            verbose=False,
        )
        sql_task = Task(config=tasks_config['consulta_sql'], agent=analyst)
        numeric_task = Task(
            config=tasks_config['ranking_numerico'],
            agent=scientist,
            context=[sql_task],
        )
        crew = Crew(
            agents=[analyst, scientist, writer],
            tasks=[sql_task, numeric_task],
            process=Process.sequential,
            verbose=False,
            max_rpm=5,
            memory=False,
        )
        crew.kickoff()
        writer_context = json.dumps({
            'dados_tool': ranking,
            'formato_factual_obrigatorio': build_deterministic_report(ranking),
            'schema': ExecutiveReport.model_json_schema(),
        }, ensure_ascii=False)

        def generate(error_message):
            description = tasks_config['relatorio_executivo']['description']
            if error_message is not None:
                description += (
                    '\nA saída anterior foi rejeitada. Corrija somente o relatório '
                    f'conforme este erro de validação:\n{error_message}'
                )
            # JSON bruto evita retries ocultos do conversor output_pydantic.
            task = Task(
                description=description,
                expected_output=tasks_config['relatorio_executivo']['expected_output'],
                agent=writer,
            )
            return writer.execute_task(task, context=writer_context)

        report, _ = generate_validated_report(generate, ranking)
        audit = verify_report_sources(report, ranking)
        return report, audit


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Gera relatório de prioridade de revisão de transações.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Gera relatório determinístico sem inicializar LLM ou servidor MCP.',
    )
    args = parser.parse_args(argv)
    load_dotenv(PROJECT_ROOT / '.env')

    if args.dry_run:
        ranking = _get_numeric_top_ten()
        report, _ = validate_report(build_deterministic_report(ranking), ranking)
        audit = verify_report_sources(report, ranking)
        print(json.dumps({
            'relatorio': report,
            'auditoria': audit,
        }, ensure_ascii=False, indent=2))
        return 0

    report, audit = _run_real_report()
    print(json.dumps({
        'relatorio': report,
        'auditoria': audit,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
