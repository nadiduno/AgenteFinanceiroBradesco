from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FactorName = Literal[
    'valor_vs_mediana_cliente',
    'cidade_nova_cliente',
    'hora_fora_padrao_cliente',
    'dispositivo_novo',
    'idade_conta_destino_dias',
    'qtd_transacoes_ultima_hora',
]


class StrictReportModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class ReportFactor(StrictReportModel):
    nome: FactorName
    rotulo: NonEmptyText
    valor: NonEmptyText


class ReportCase(StrictReportModel):
    id_transacao: int = Field(ge=1)
    posicao: int = Field(ge=1)
    resumo: Annotated[NonEmptyText, Field(max_length=500)]
    fatores: list[ReportFactor] = Field(min_length=3, max_length=3)


class ExecutiveReport(StrictReportModel):
    casos: list[ReportCase] = Field(min_length=1, max_length=20)
    recomendacoes: list[NonEmptyText] = Field(min_length=1)
    limitacoes: list[NonEmptyText] = Field(min_length=1)
    degradado: bool = False
