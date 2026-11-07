import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

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


class ReportFactor(BaseModel):
    nome: str
    rotulo: str
    valor: str


class ReportCase(BaseModel):
    id_transacao: int
    posicao: int
    resumo: str
    fatores: list[ReportFactor]


class ExecutiveReport(BaseModel):
    casos: list[ReportCase]
    recomendacoes: list[str]
    limitacoes: list[str]


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
    from crewai import LLM

    return LLM(model=model_name, temperature=0.1, max_tokens=2048, timeout=60)


def _validate_llm_report(candidate, ranking):
    report = candidate.model_dump()
    source_report = build_deterministic_report(ranking)
    candidate_cases = report.get('casos', [])
    source_cases = source_report['casos']
    if len(candidate_cases) != len(source_cases):
        raise ValueError('O Redator deve retornar exatamente dez casos.')

    for candidate_case, source_case in zip(candidate_cases, source_cases):
        if (
            candidate_case['id_transacao'] != source_case['id_transacao']
            or candidate_case['posicao'] != source_case['posicao']
        ):
            raise ValueError('O Redator alterou um ID ou posição do ranking.')
        candidate_factors = {
            (factor['nome'], factor['rotulo'], factor['valor'])
            for factor in candidate_case['fatores']
        }
        source_factors = {
            (factor['nome'], factor['rotulo'], factor['valor'])
            for factor in source_case['fatores']
        }
        if candidate_factors != source_factors:
            raise ValueError('O Redator alterou ou inventou features ou valores.')

    for item in report['recomendacoes'] + report['limitacoes']:
        if any(character.isdigit() for character in item):
            raise ValueError(
                'Recomendações e limitações não podem introduzir IDs, valores ou datas.'
            )
    rendered = json.dumps(report, ensure_ascii=False).casefold()
    if 'probabilidade de fraude' in rendered:
        raise ValueError('O relatório não pode usar a expressão proibida.')

    source_report['recomendacoes'] = report['recomendacoes']
    source_report['limitacoes'] = report['limitacoes']
    return source_report


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
        executive_task = Task(
            config=tasks_config['relatorio_executivo'],
            agent=writer,
            context=[sql_task, numeric_task],
            output_pydantic=ExecutiveReport,
        )
        crew = Crew(
            agents=[analyst, scientist, writer],
            tasks=[sql_task, numeric_task, executive_task],
            process=Process.sequential,
            verbose=False,
            max_rpm=5,
            memory=False,
        )
        result = crew.kickoff()

    generated = result.tasks_output[-1].pydantic
    if not isinstance(generated, ExecutiveReport):
        raise RuntimeError(
            'O Redator não produziu a saída output_pydantic esperada; '
            'nenhum relatório será apresentado como válido.'
        )
    report = _validate_llm_report(generated, ranking)
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
        report = build_deterministic_report(ranking)
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
