# Crivo

Avaliação reproduzível de respostas estruturadas de IA, com métricas, matriz de confusão e um gate que reprova resultados abaixo do limite escolhido.

**Problema:** mudar um prompt pode melhorar um exemplo e piorar outros. Este projeto versiona casos, executa previsões e transforma os resultados em critérios verificáveis.

## Tarefa avaliada

Classificar mensagens fictícias nas categorias `compras`, `financeiro`, `comercial` e `outros`, além de decidir se há necessidade de revisão. O conjunto tem 12 casos, incluindo negações, ambiguidade e uma tentativa de alterar as instruções do classificador.

## Executar sem API

Python 3.11 ou superior. Sem dependências externas.

```powershell
python app.py replay --output outputs/replay
```

O replay usa previsões sintéticas que contêm dois erros deliberados. A acurácia exata é 10/12, abaixo do gate padrão de 85%, e o comando termina com código 1. Isso demonstra uma reprovação; não representa a qualidade medida de um modelo real.

Arquivos gerados: `report.json`, `report.md` e `predictions.jsonl`.

## Avaliar um modelo real

Configure as variáveis de ambiente e execute explicitamente:

```powershell
python app.py live --prompt explicit --output outputs/explicit
python app.py live --prompt minimal --output outputs/minimal
```

Cada comando gera uma chamada por caso. O limite padrão é 20 casos; aumentar o limite exige `--max-cases`. Os relatórios registram o uso de tokens retornado pela API. Não é calculado custo monetário sem uma tabela de preços correspondente ao modelo.

## Critérios de qualidade

- Acurácia exata: categoria e sinalização de revisão precisam corresponder à referência.
- Acurácia de categoria: considera apenas a categoria, mas exige saída com formato válido.
- Macro F1: média das categorias presentes nas referências, com erros de formato contados como falsos negativos.
- Formato válido: tipos, campos e enumeração esperados.
- Previsões ausentes contam como falha; IDs duplicados ou desconhecidos são rejeitados.
- Retorno 0: gate aprovado. Retorno 1: gate reprovado. Retorno 2: configuração, entrada ou saída inválida.

## Decisões e limites

Os critérios são determinísticos, sem um segundo modelo atuando como juiz. O conjunto pequeno serve à demonstração de regressões; não sustenta conclusões gerais de qualidade. Os rótulos esperados são escolhas explícitas de uma taxonomia fictícia. Em uso real, seriam necessários mais casos, revisão dos rótulos e avaliação representativa do processo.

## O que demonstra

Automação de avaliações, comparação de prompts, métricas, testes de regressão, tratamento de falhas de modelo e qualidade como complemento ao desenvolvimento de IA.

## IA real e demonstrações locais

A integração real usa a Responses API da OpenAI e JSON Schema. Configure `OPENAI_API_KEY` e `OPENAI_MODEL` nas variáveis de ambiente, escolhendo um modelo compatível com Structured Outputs. Não há modelo fixo ou chave incluída. Arquivos `.env` não são carregados automaticamente.

Os modos locais não chamam um modelo e não devem ser apresentados como resultados de IA. Os testes usam respostas simuladas de rede. A integração real ainda precisa ser validada com uma chave e um modelo disponíveis na sua conta.

No modo OpenAI, o conteúdo enviado como entrada é transmitido ao serviço. Use os dados fictícios incluídos para começar. A chave fica no ambiente e não entra no código, nos relatórios ou nos logs. A configuração `store=false` é enviada à API; ela não substitui a política de tratamento de dados do provedor.

## Testes

```powershell
python -m unittest discover -s tests -v
```

O fluxo de testes do GitHub Actions usa apenas casos locais e respostas simuladas, sem credenciais de IA. O workflow está preparado, mas a execução no GitHub depende da publicação do repositório.

## Contexto do projeto

Projeto de portfólio de João Carlos Minozzi, desenvolvido com assistência de IA. Os exemplos são fictícios e não representam implantação em clientes ou uso em produção. As decisões e limitações estão documentadas para apoiar estudo, revisão e evolução do código.

Referência de integração: [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
