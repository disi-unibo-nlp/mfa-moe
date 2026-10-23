"""The frozen reader's parsing of a raw judge output into a class label.

`driver.lfm_annotate` builds a bare `Reader` (no `__init__`) and sets `annotator`, `renderer` and
`tokenizer` on it before calling `parse`.
"""
from moe_exp.io import CLASSES


class Reader:
    def parse(self, row, raw):
        from vllm.reasoning import ReasoningParserManager
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        a = self.annotator
        parser = ReasoningParserManager.get_reasoning_parser('qwen3')(self.tokenizer)
        request = ChatCompletionRequest(model=a['config']['judge_model'],
            messages=self.renderer.request('render', inputs=row['inputs'])['conversation'],
            reasoning_effort=a['config']['reasoning_effort'])
        text = self.tokenizer.decode(raw['token_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        _, content = parser.extract_reasoning(text, request)
        parsed = self.renderer.request('parse', content=content or '') if raw['finish_reason'] == 'stop' else {}
        label = parsed.get('label')
        return {'status': 'complete' if label in CLASSES else 'unknown',
                'label': label if label in CLASSES else None, 'raw_sha256': raw['sha256'],
                'finish_reason': raw['finish_reason'], 'generated_tokens': len(raw['token_ids'])}
