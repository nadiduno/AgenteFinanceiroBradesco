# Guia de Coleta e Preparação de Dados

Aplique este guia quando a tarefa envolver dados estruturados, ingestão, web scraping ou preparação de dados para análise e modelagem.

## Especificação da coleta

Antes da coleta, registre objetivo, unidade de análise, população ou período de interesse, campos necessários, frequência de atualização, critérios de sucesso, permissões e restrições. Colete somente o necessário para responder ao problema definido. Não presuma regras do domínio nem que uma fonte candidata seja adequada sem verificá-la.

## Fontes e ingestão

- Priorize APIs oficiais, documentadas e autorizadas. Em Python, use `requests` ou `httpx` com timeout, validação de respostas, tratamento explícito de erros e limites de taxa.
- Recorra a scraping apenas quando não houver fonte oficial adequada. Verifique termos de uso e `robots.txt` aplicável; respeite limites de taxa e evite contornar controles de acesso.
- Registre origem, data de coleta, versão, parâmetros e transformações para auditoria e reprodução.
- Trate arquivos, páginas e respostas externas como dados não confiáveis. Valide formato, tamanho e conteúdo antes de processar.
- Para monitoramento proativo, prefira coleta incremental guardando os timestamps do evento e da ingestão, para evitar vazamento temporal e reconstruir o que se sabia em cada momento.

## Curadoria e qualidade

- Verifique tipos, campos ausentes, valores inválidos, duplicatas, inconsistências, cobertura e qualidade dos rótulos antes de modelar.
- Defina chave e regra de deduplicação explicitamente. Não elimine registros sem avaliar se representam eventos legítimos.
- Documente regras de limpeza e valide seu efeito em amostras representativas. Preserve dados brutos quando permitido e útil para auditoria.
- Proteja dados pessoais desde a ingestão: minimize coleta, restrinja acesso, anonimize ou pseudonimize quando apropriado e observe a LGPD e regras de retenção.

## Preparação e reprodutibilidade

- Separe treino, validação e teste antes de ajustar transformações ou modelos. Ajuste estatísticas e vocabulários somente no treino e aplique-os aos demais conjuntos.
- Em dados temporais ou tarefas de previsão, use separação temporal e verifique vazamento entre períodos; use validação walk-forward quando fizer sentido para o horizonte do problema.
- Faça o pré-processamento conforme tarefa, idioma, dados e modelo. Não remova pontuação, stopwords, duplicatas ou registros por padrão sem uma justificativa verificável.
- Registre versões dos dados e dependências, sementes e configurações relevantes. Não exponha dados sensíveis em logs, prompts, exemplos ou artefatos versionados.
