# AgenteFinanceiroBradesco
Sistema de Agentes de IA autônomos e orquestrados com CrewAI para apoio à tomada de decisão financeira e detecção de anomalias em transações bancárias. Integração de servidor MCP (Model Context Protocol) em Python para consumo de bases de dados relacionais via SQL, exposto com interface em Lovable 

## API experimental (Fase 6)

Instale `requirements.txt` e execute `python -m uvicorn src.api.app:app`.
`torch` e `transformers` não são necessários para o modelo numérico padrão;
para a comparação experimental Hugging Face, instale `requirements-hf.txt`.

Preencha o `.env` a partir de `.env.example`:

- `API_ACCESS_KEY`: chave própria da API, **diferente** de `GEMINI_API_KEY`.
  Sem configuração, `/analisar` fica indisponível (503); `/health` é público.
- `CORS_ORIGINS`: array JSON de origens exatas, por exemplo
  `["http://localhost:3000"]`, sem barra final ou wildcard. Padrão `[]`.
  Requisições com origem não permitida, inclusive preflight, recebem 403.
- `DATABASE_URL`: arquivo SQLite local. Se ausente, a inicialização gera
  5.000 transações sintéticas com semente 42; bancos existentes são preservados.
- `ALLOW_LLM=false`: padrão sem provedor. Só `ALLOW_LLM=true` **e**
  `use_llm=true` habilitam LLM; opt-in bloqueado recebe 403, sem fallback silencioso.
- `LLM_MAX_CALLS_PER_MINUTE=5`: janela móvel de 60 segundos compartilhada
  pelas requisições do mesmo processo; conta chamadas nativas, não relatórios.
- `LLM_TIMEOUT_SECONDS=60`: prazo da execução LLM (positivo, até 300 segundos)
  e timeout do SDK. A API não faz retries HTTP do provedor.

### Endpoints

- `GET /health`: `{"status":"ok"}`.
- `POST /analisar`: cabeçalho `X-API-Key` obrigatório; JSON
  `{"top_n":10,"use_llm":false}` (campos opcionais; `top_n` inteiro de 1 a 20).
  Devolve diretamente o relatório validado da Fase 5, incluindo `degradado`;
  não devolve auditoria, descrições livres nem rótulos. A CLI continua top-10.

Entrada inválida recebe 422; chave ausente/incorreta, 401; limite LLM, 429;
timeout, 504; falha interna, 500. Erros são genéricos, com `request_id`
também no cabeçalho `X-Request-ID` e nos logs. Logs de dependências durante
a requisição são reduzidos a evento/nível e ID, sem payloads ou traceback.

Modelo numérico e crew executam em threadpool. Um timeout devolve 504 e
impede novas chamadas daquele trabalho, mas não mata uma thread ou chamada
HTTP já em curso; esta permanece sujeita ao timeout do SDK. Para múltiplos
workers/réplicas, o limite por processo não constitui uma quota global:
use um limitador distribuído antes de escalar. Geração inicial simultânea
do mesmo banco por múltiplos workers não é suportada; inicialize com um worker.
Se a crew encapsular ou suprimir o erro de prazo, a resposta continua 504,
sem apresentar o resultado daquele trabalho como sucesso.

Os scores são heurísticos em dados sintéticos, não probabilidades nem evidência
de eficácia em fraude real. O ranking usa a base inteira para revisão, não
avaliação de desempenho. Não use o relatório para decisões financeiras reais.

Testes offline: `python -m unittest discover -s tests -v`.
