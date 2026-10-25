import json
import os
import unittest
from unittest.mock import patch

import httpx

from ds_harness.config import LLMConfig, StrictModel
from ds_harness.errors import LLMError
from ds_harness.llm import ChatClient


class Answer(StrictModel):
    answer: int


def response(content, status=200, **extra):
    return httpx.Response(status, json={"choices": [{"message": {"content": content},
                                                   "finish_reason": "stop"}], **extra})


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"DS_LLM_BASE_URL": "https://inference.example.test/v1",
                                          "DS_LLM_MODEL": "test-model", "DS_LLM_API_KEY": "SECRET"})
        self.env.start()
        self.events = []

    def tearDown(self):
        self.env.stop()

    def client(self, handler, **kwargs):
        return ChatClient(LLMConfig(**kwargs), self.events.append, httpx.MockTransport(handler))

    def test_compatible_request_and_redacted_logs(self):
        def handler(request):
            self.assertEqual(str(request.url), "https://inference.example.test/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer SECRET")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "test-model")
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            return response('{"answer":42}', usage={"total_tokens": 20})
        result = self.client(handler).structured("Return JSON", {}, Answer)
        self.assertEqual(result.answer, 42)
        self.assertNotIn("SECRET", str(self.events))
        self.assertEqual(self.events[-1]["usage"]["total_tokens"], 20)

    def test_repairs_invalid_json(self):
        replies = iter(["invalid", '{"answer":"wrong"}', '{"answer":7}'])
        client = self.client(lambda _: response(next(replies)))
        self.assertEqual(client.structured("Return JSON", {}, Answer).answer, 7)
        self.assertEqual(client.calls, 3)

    def test_budget_includes_retries(self):
        client = self.client(lambda _: httpx.Response(429), max_calls=2)
        with patch("ds_harness.llm.time.sleep"), self.assertRaisesRegex(LLMError, "budget"):
            client.complete([{"role": "user", "content": "JSON"}])
        self.assertEqual(client.calls, 2)

    def test_unauthorized_response_not_retried_or_logged(self):
        client = self.client(lambda _: httpx.Response(401, text="SECRET"))
        with self.assertRaises(LLMError) as context:
            client.complete([])
        self.assertNotIn("SECRET", str(context.exception))
        self.assertEqual(client.calls, 1)

    def test_redirect_is_rejected(self):
        client = self.client(lambda _: httpx.Response(302, headers={"location": "https://other.test"}))
        with self.assertRaises(LLMError):
            client.complete([])
        self.assertEqual(client.calls, 1)

    def test_plaintext_remote_endpoint_rejected(self):
        with patch.dict(os.environ, {"DS_LLM_BASE_URL": "http://remote.example.test/v1"}):
            client = self.client(lambda _: response("{}"))
            with self.assertRaisesRegex(LLMError, "HTTPS"):
                client.complete([])
        self.assertEqual(client.calls, 0)

    def test_local_endpoint_and_compatibility_switches(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertNotIn("response_format", payload)
            self.assertIn("max_completion_tokens", payload)
            self.assertEqual(request.headers["api-key"], "SECRET")
            return response('{"answer":1}')
        with patch.dict(os.environ, {"DS_LLM_BASE_URL": "http://127.0.0.1:8080/v1"}):
            client = self.client(handler, json_mode=False, auth_header="api-key",
                                 token_parameter="max_completion_tokens")
            self.assertEqual(client.structured("JSON", {}, Answer).answer, 1)

    def test_truncated_response_is_rejected(self):
        client = self.client(lambda _: httpx.Response(200, json={"choices": [
            {"message": {"content": "{}"}, "finish_reason": "length"}]}))
        with self.assertRaisesRegex(LLMError, "truncated"):
            client.complete([])

    def test_network_timeouts_are_bounded(self):
        def handler(request):
            raise httpx.ReadTimeout("secret internal URL", request=request)
        client = self.client(handler, retries=1)
        with patch("ds_harness.llm.time.sleep"), self.assertRaises(LLMError) as context:
            client.complete([])
        self.assertEqual(client.calls, 2)
        self.assertNotIn("secret", str(context.exception))


if __name__ == "__main__":
    unittest.main()
