# Memória do Agente DataScience

## Status atual

- **Nome:** DataScience.
- **Projeto:** AgenteFinanceiroBradesco.
- **Estado:** Fase 1 em andamento (Estrutura e Dados Sintéticos).
- **Arquivos criados:** `.gitignore`, `.env.example`, `requirements.txt` (atualizado), `src/utils/db_gen.py`.
- **Decisões:** Uso de SQLite para dados sintéticos, estrutura modular para MCP e CrewAI.

## Próximo passo

- Executar `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- Executar `.venv\Scripts\python.exe src/utils/db_gen.py` para gerar `data/transacoes.db`.
- Prosseguir para a Fase 2: Servidor MCP SQL.

## Log de sessões

- 2026-11-02: nomeado o agente DataScience; generalizado o contrato, definidos guardrails e criados guias técnicos sob demanda. Nenhum domínio ou regra de negócio global foi presumido.
- 2026-11-02: complementados os guias de coleta, NLP/RAG e multimodalidade/entrega; a tabela de roteamento correspondente já consta em `GEMINI.md`.
- 2026-11-02: criado `docs/SPEC_TEMPLATE.md` como modelo de especificação por projeto; as instruções gerais agora orientam seu uso e a atualização de `memory.md` quando o escopo mudar.
- 2026-11-02: adicionados seis testes automatizados para validar estrutura, critérios, opcionalidade dos entregáveis e ligação do template às instruções; `python -m unittest discover -v` passou (6 testes).
- 2026-11-02: Início do projeto AgenteFinanceiroBradesco. Aprovada estrutura de arquivos (Fase 0). Implementados arquivos base e script de geração de dados sintéticos (Fase 1).
