# Guia de Visualização, Serviços e Entrega

Aplique este guia quando a tarefa envolver comunicação visual, disponibilização de serviços, implantação, monitoramento ou recursos multimodais.

## Visualização e comunicação

- Use títulos conclusivos que expressem o principal achado, unidades e rótulos claros, escalas honestas e cores com propósito.
- Destaque comparações sem distorcer os dados. Garanta legibilidade e acessibilidade, inclusive para pessoas com deficiência de visão de cores.
- Comunique premissas, limitações, incertezas e implicações práticas junto aos resultados.
- Em problemas preditivos, apresente a linha de base, métrica e justificativa, calibração e incerteza, antecedência útil, falsos alarmes e desempenho por segmento quando aplicável.

## Serviços e implantação

- Para serviços HTTP, considere APIs robustas, por exemplo FastAPI, se fizerem sentido para requisitos de integração e operação. Valide entradas, aplique autenticação e autorização apropriadas, limites de uso e tratamento seguro de erros.
- Versione código, dados ou referências a dados, modelos, prompts e configurações. Documente como reproduzir avaliações, atualizar componentes e reverter uma implantação.
- Antes de produção, verifique critérios de aceite, segurança, privacidade, qualidade, latência, custo, monitoramento, plano de manutenção e rollback.
- Monitore erros, qualidade, latência, custos, feedback e mudanças na distribuição de dados. Inclua revisão humana para decisões irreversíveis ou de alto impacto.

## Multimodalidade e serviços externos

- Use imagem, áudio e vídeo somente quando forem pertinentes ao objetivo e houver dados, permissões, infraestrutura e critérios de avaliação adequados.
- Avalie plataformas de geração ou controle de ativos visuais e vídeos conforme disponibilidade, termos, permissões, privacidade, custos e integração técnica. Não as trate como dependências obrigatórias.
- Antes de enviar dados a serviços externos, confirme termos vigentes, retenção, uso para treinamento, localização, permissões e custo. Minimize ou anonimize os dados; não envie segredos ou dados pessoais sem autorização e base legal apropriadas.
- Não presuma disponibilidade, integração, licença ou segurança de um fornecedor específico. Avalie a solução e seus riscos antes de adotá-la.
