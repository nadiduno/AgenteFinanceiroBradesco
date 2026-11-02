# SPEC — [nome do projeto]

> Uma spec por projeto: curta, verificável e baseada em requisitos confirmados. Copie este arquivo para `SPEC.md` no projeto correspondente. O agente deve lê-la antes de implementar e atualizá-la quando o escopo mudar, registrando a decisão em `memory.md`.
>
> Status: rascunho | validada | em implementação | concluída

## 1. Contexto e objetivo

- **Problema / evento a prevenir:**
- **Quem usa e que decisão toma com o resultado:**
- **Horizonte (lead time), se aplicável:**
- **Ação disparada pelo alerta, se aplicável:**
- **Custo de erro:** falso negativo = ; falso positivo =

## 2. Dados

| Fonte | Local | Período | Sensível? (LGPD) | Observações |
|---|---|---|---|---|
| | | | | |

## 3. Critérios de aceitação

- **Métrica principal e meta:**
- **Baseline a superar:**
- **Validação:** estratégia e justificativa (por exemplo, temporal/walk-forward; gap ≥ horizonte quando aplicável).
- **Limite de falsos alarmes por semana, se aplicável:**
- **Calibração e intervalo de incerteza, se aplicável:**

## 4. Entregáveis

Marque apenas o que fizer parte do escopo confirmado; remova ou adapte itens não aplicáveis.

- [ ] `solutions/<projeto>/solution.py`: solução executável de ponta a ponta
- [ ] `solutions/<projeto>/notebook.ipynb`: EDA, validação e conclusões
- [ ] `apps/api`: endpoints FastAPI documentados no Swagger (`/docs`)
- [ ] `apps/web`: telas React que consomem a API
- [ ] Testes e README com instruções de execução
- [ ] Outros:

## 5. Contrato da API

Preencher se houver API no escopo. Documentar schemas, exemplos e erros esperados.

| Método | Rota | Entrada (schema) | Saída (schema) | Erros |
|---|---|---|---|---|
| GET | `/health` | — | status | — |
| POST | `/api/v1/predict` | | | 422, 503 |

## 6. Interface (React)

Preencher se houver interface no escopo.

- **Telas e fluxos principais:**
- **Estados a tratar:** carregando, erro, vazio, baixa confiança / sem evidência

## 7. Requisitos não funcionais

- **Latência alvo:**
- **Custo:**
- **Privacidade:** não registrar PII em logs; outros requisitos:
- **Disponibilidade:**
- **Outros:**

## 8. Fora de escopo

-

## 9. Riscos e hipóteses em aberto

- `[HIPÓTESE]`

## 10. Definition of Done

- [ ] Critérios da seção 3 atingidos nas verificações definidas
- [ ] Contrato da API completo e com exemplos, se houver API
- [ ] Testes passando e lint limpo, conforme ferramentas do projeto
- [ ] README com instruções de execução, se aplicável
- [ ] `memory.md` atualizado com decisões de escopo relevantes
