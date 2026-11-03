# Memória do Agente DataScience

## Status atual

- **Nome:** DataScience.
- **Projeto:** AgenteFinanceiroBradesco.
- **Estado:** Fase 1 concluída (Estrutura e Dados Sintéticos).
- **Arquivos:** `.gitignore`, `.env.example`, `requirements.txt`, `src/utils/db_gen.py` e `data/transacoes.db` (gerado localmente e ignorado pelo Git).
- **Decisões:** Dados sintéticos em SQLite; rótulos de fraude isolados em `rotulos_fraude` para avaliação, fora de `transacoes`.

## Próximo passo

- Fase 1 validada; dependências já estão instaladas no `.venv` e não devem ser reinstaladas.
- Prosseguir para a Fase 2: servidor MCP SQL somente leitura, sem acesso dos agentes a `rotulos_fraude`.

## Log de sessões

- 2026-11-02: nomeado o agente DataScience; generalizado o contrato, definidos guardrails e criados guias técnicos sob demanda. Nenhum domínio ou regra de negócio global foi presumido.
- 2026-11-02: complementados os guias de coleta, NLP/RAG e multimodalidade/entrega; a tabela de roteamento correspondente já consta em `GEMINI.md`.
- 2026-11-02: criado `docs/SPEC_TEMPLATE.md` como modelo de especificação por projeto; as instruções gerais agora orientam seu uso e a atualização de `memory.md` quando o escopo mudar.
- 2026-11-02: adicionados seis testes automatizados para validar estrutura, critérios, opcionalidade dos entregáveis e ligação do template às instruções; `python -m unittest discover -v` passou (6 testes).
- 2026-11-02: Início do projeto AgenteFinanceiroBradesco. Aprovada estrutura de arquivos (Fase 0). Implementados arquivos base e script de geração de dados sintéticos (Fase 1).
- 2026-11-02: concluída a Fase 1; gerado SQLite com 500 clientes, 5.000 transações e tabela separada com 5.000 rótulos (100 positivos); conferidas as colunas e a integridade referencial.
