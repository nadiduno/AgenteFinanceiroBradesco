REGRAS DE EXECUÇÃO:
- Ambiente: Windows + PowerShell. Não use comandos do cmd (dir /s /b).
- As dependências JÁ estão instaladas no .venv. Não rode pip install.
  Use sempre o Python do venv: .venv\Scripts\python.exe
- Crie os arquivos direto no projeto; não apenas mostre o código no chat.
- Use dados SINTÉTICOS em SQLite (data/transacoes.db). Nada de dados reais.
- Nunca coloque chaves no código: use .env. Crie .env.example e .gitignore
  (.venv/, .env, data/*.db).
- Na árvore de arquivos, use a pasta mcp_servers/ no lugar de mcp/
  (evita conflito com o pacote 'mcp').

ENTREGA EM FASES (pare ao fim de cada uma e espere eu escrever "continuar"):
Fase 0: proponha a árvore de arquivos. Não crie nada ainda.
Fase 1: estrutura de pastas, .env.example, .gitignore e script que gera o SQLite sintético.
Fase 2: servidor MCP somente leitura (describe_schema e run_select, só SELECT).
Fase 3: Isolation Forest + pipeline Hugging Face em CPU.
Fase 4: agentes e tasks CrewAI (config/agents.yaml e tasks.yaml) e main_crew.py.
Fase 5: schemas Pydantic e guardrails anti-alucinação.
Fase 6: API FastAPI (/analisar, /health) com CORS para o Lovable.
Fase 7: README.md completo.
