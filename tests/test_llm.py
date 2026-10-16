from io import BytesIO
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from llm import ModelError,structured_response

SCHEMA={'type':'object','properties':{},'required':[],'additionalProperties':False}

def fake_response(value):
    return {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(value)}]}],
            'usage':{'input_tokens':12,'output_tokens':3}}

class ClientTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict('os.environ',{'OPENAI_API_KEY':'unit-test-key','OPENAI_MODEL':'unit-test-model'},clear=True)
        self.env.start();self.addCleanup(self.env.stop)
    def run_response(self,body):
        return structured_response('rules',{'test':True},SCHEMA,'test',opener=lambda *a,**k:BytesIO(json.dumps(body).encode()))
    def test_request_has_schema_and_no_storage(self):
        def opener(request,timeout):
            payload=json.loads(request.data)
            self.assertFalse(payload['store']);self.assertTrue(payload['text']['format']['strict'])
            self.assertEqual(request.full_url,'https://api.openai.com/v1/responses')
            self.assertEqual(request.get_method(),'POST');self.assertEqual(timeout,45)
            return BytesIO(json.dumps(fake_response({'ok':True})).encode())
        result,usage=structured_response('rules',{},SCHEMA,'test',opener=opener)
        self.assertEqual(result,{'ok':True});self.assertEqual(usage['input_tokens'],12)
    def test_missing_key_does_not_call_network(self):
        with patch.dict('os.environ',{},clear=True),self.assertRaises(ModelError):
            structured_response('',{},SCHEMA,'test',opener=lambda *a: self.fail('não pode chamar rede'))
    def test_incomplete_response_is_rejected(self):
        with self.assertRaises(ModelError):self.run_response({'status':'incomplete','output':[]})
    def test_refusal_is_rejected(self):
        with self.assertRaises(ModelError):self.run_response({'status':'completed','output':[{'content':[{'type':'refusal'}]}]})
    def test_invalid_json_is_rejected(self):
        body=fake_response({});body['output'][0]['content'][0]['text']='not json'
        with self.assertRaises(ModelError):self.run_response(body)
    def test_no_message_is_rejected(self):
        with self.assertRaises(ModelError):self.run_response({'status':'completed','output':[]})
    def test_http_error_does_not_expose_body_or_key(self):
        def failing(*a,**k):raise HTTPError('https://api.openai.com',401,'secret-body',{},BytesIO(b'secret'))
        with self.assertRaises(ModelError) as caught:
            structured_response('',{},SCHEMA,'test',opener=failing)
        self.assertNotIn('secret',str(caught.exception));self.assertNotIn('unit-test-key',str(caught.exception))

if __name__=='__main__':unittest.main()
