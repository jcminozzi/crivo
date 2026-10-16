"""Small Responses API adapter. No paid calls occur on import or in tests."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelError(RuntimeError):
    pass


def structured_response(instructions, data, schema, name, *, model=None, opener=None):
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    model = model or os.environ.get('OPENAI_MODEL', '').strip()
    if not key or not model:
        raise ModelError('Configure OPENAI_API_KEY e OPENAI_MODEL para usar IA real.')
    payload = {
        'model': model, 'instructions': instructions,
        'input': json.dumps(data, ensure_ascii=False), 'store': False,
        'max_output_tokens': 1600,
        'text': {'format': {'type': 'json_schema', 'name': name,
                            'schema': schema, 'strict': True}},
    }
    request = Request('https://api.openai.com/v1/responses',
                      data=json.dumps(payload).encode('utf-8'),
                      headers={'Authorization': 'Bearer ' + key,
                               'Content-Type': 'application/json'}, method='POST')
    try:
        with (opener or urlopen)(request, timeout=45) as response:
            body = json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        # Do not print request/response bodies: they can contain private documents.
        raise ModelError(f'API retornou HTTP {exc.code}; confira acesso, quota e modelo.') from None
    except (URLError, TimeoutError, OSError, UnicodeError, ValueError):
        raise ModelError('Falha de conexão ou resposta inválida da API.') from None
    if not isinstance(body, dict) or body.get('status') != 'completed':
        raise ModelError('Resposta da IA não concluída; nenhuma saída foi aceita.')
    texts = []
    output = body.get('output', [])
    if not isinstance(output, list):
        raise ModelError('Estrutura de resposta inválida.')
    for item in output:
        if not isinstance(item, dict):
            raise ModelError('Estrutura de resposta inválida.')
        for content in item.get('content', []):
            if not isinstance(content, dict):
                raise ModelError('Estrutura de conteúdo inválida.')
            if content.get('type') == 'refusal':
                raise ModelError('A solicitação foi recusada pelo modelo.')
            if content.get('type') == 'output_text':
                texts.append(content.get('text', ''))
    if not texts:
        raise ModelError('Resposta concluída sem conteúdo estruturado.')
    try:
        value = json.loads(''.join(texts))
    except (TypeError, ValueError):
        raise ModelError('A resposta não contém JSON válido.') from None
    if not isinstance(value, dict):
        raise ModelError('A saída deve ser um objeto JSON.')
    usage = body.get('usage') or {}
    if not isinstance(usage, dict):
        usage = {}
    return value, {k: usage.get(k, 0) for k in ('input_tokens', 'output_tokens')}
