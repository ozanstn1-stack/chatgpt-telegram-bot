from __future__ import annotations
import datetime
import logging
import os
import json
import httpx
import io
from PIL import Image
import tiktoken
import openai
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from utils import is_direct_result, encode_image, decode_image
from plugin_manager import PluginManager

# DeepSeek model sabiti
MODEL_NAME = "deepseek-v4-pro"

# DeepSeek uyumlu max token
def default_max_tokens(model: str) -> int:
    return 4096

def are_functions_available(model: str) -> bool:
    return True

# Dil dosyası
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
translations_file_path = os.path.join(parent_dir_path, 'translations.json')
with open(translations_file_path, 'r', encoding='utf-8') as f:
    translations = json.load(f)

def localized_text(key, bot_language):
    try:
        return translations[bot_language][key]
    except KeyError:
        logging.warning(f"No translation for '{key}' in '{bot_language}'")
        return translations.get('en', {}).get(key, key)

class OpenAIHelper:
    def __init__(self, config: dict, plugin_manager: PluginManager):
        http_client = httpx.AsyncClient(proxy=config['proxy']) if 'proxy' in config else None
        self.client = openai.AsyncOpenAI(
            api_key=config['api_key'],
            base_url=os.environ.get('OPENAI_API_BASE', 'https://api.deepseek.com/v1'),
            http_client=http_client
        )
        self.config = config
        self.plugin_manager = plugin_manager
        self.conversations: dict[int, list] = {}
        self.conversations_vision: dict[int, bool] = {}
        self.last_updated: dict[int, datetime.datetime] = {}

    def get_conversation_stats(self, chat_id: int) -> tuple[int, int]:
        if chat_id not in self.conversations:
            self.reset_chat_history(chat_id)
        return len(self.conversations[chat_id]), self.__count_tokens(self.conversations[chat_id])

    async def get_chat_response(self, chat_id: int, query: str) -> tuple[str, str]:
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query)
        if self.config.get('enable_functions') and not self.conversations_vision.get(chat_id):
            response, plugins_used = await self.__handle_function_call(chat_id, response)
            if is_direct_result(response):
                return response, '0'
        answer = response.choices[0].message.content.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)
        return answer, str(response.usage.total_tokens)

    async def get_chat_response_stream(self, chat_id: int, query: str):
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query, stream=True)
        if self.config.get('enable_functions') and not self.conversations_vision.get(chat_id):
            response, plugins_used = await self.__handle_function_call(chat_id, response, stream=True)
            if is_direct_result(response):
                yield response, '0'
                return
        answer = ''
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer, 'not_finished'
        answer = answer.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)
        yield answer, str(self.__count_tokens(self.conversations[chat_id]))

    @retry(reraise=True, retry=retry_if_exception_type(openai.RateLimitError), wait=wait_fixed(20), stop=stop_after_attempt(3))
    async def __common_get_chat_response(self, chat_id: int, query: str, stream=False):
        bot_language = self.config.get('bot_language', 'en')
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)
            self.last_updated[chat_id] = datetime.datetime.now()
            self.__add_to_history(chat_id, role="user", content=query)

            # Konuşma çok uzadıysa özetle
            token_count = self.__count_tokens(self.conversations[chat_id])
            if token_count + self.config['max_tokens'] > 4096 or len(self.conversations[chat_id]) > self.config['max_history_size']:
                try:
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.__add_to_history(chat_id, role="user", content=query)
                except Exception:
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            common_args = {
                'model': MODEL_NAME,
                'messages': self.conversations[chat_id],
                'temperature': self.config.get('temperature', 0.7),
                'n': self.config.get('n_choices', 1),
                'max_tokens': self.config.get('max_tokens', 4096),
                'presence_penalty': self.config.get('presence_penalty', 0),
                'frequency_penalty': self.config.get('frequency_penalty', 0),
                'stream': stream
            }
            if self.config.get('enable_functions') and not self.conversations_vision.get(chat_id):
                functions = self.plugin_manager.get_functions_specs()
                if functions:
                    common_args['functions'] = functions
                    common_args['function_call'] = 'auto'
            return await self.client.chat.completions.create(**common_args)
        except openai.BadRequestError as e:
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def __handle_function_call(self, chat_id, response, stream=False, times=0, plugins_used=()):
        function_name = ''
        arguments = ''
        if stream:
            async for item in response:
                if item.choices:
                    first = item.choices[0]
                    if first.delta and first.delta.function_call:
                        function_name += first.delta.function_call.name or ''
                        arguments += first.delta.function_call.arguments or ''
                    elif first.finish_reason == 'function_call':
                        break
                    else:
                        return response, plugins_used
                else:
                    return response, plugins_used
        else:
            first = response.choices[0] if response.choices else None
            if first and first.message.function_call:
                function_name = first.message.function_call.name or ''
                arguments = first.message.function_call.arguments or ''
            else:
                return response, plugins_used

        function_response = await self.plugin_manager.call_function(function_name, self, arguments)
        plugins_used += (function_name,)
        if is_direct_result(function_response):
            self.__add_function_call_to_history(chat_id, function_name, json.dumps({'result': 'Done'}))
            return function_response, plugins_used
        self.__add_function_call_to_history(chat_id, function_name, function_response)
        resp = await self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=self.conversations[chat_id],
            functions=self.plugin_manager.get_functions_specs(),
            function_call='auto' if times < self.config.get('functions_max_consecutive_calls', 5) else 'none',
            stream=stream
        )
        return await self.__handle_function_call(chat_id, resp, stream, times + 1, plugins_used)

    async def generate_image(self, prompt: str) -> tuple[str, str]:
        return "🖼️ Resim oluşturma şu anda desteklenmiyor.", "0"

    async def generate_speech(self, text: str) -> tuple[any, int]:
        raise Exception("Ses sentezi desteklenmiyor.")

    async def transcribe(self, filename):
        raise Exception("Ses tanıma desteklenmiyor.")

    # GÖRSEL YORUMLAMA TAMAMEN DEVRE DIŞI
    async def interpret_image(self, chat_id, fileobj, prompt=None):
        return "🖼️ Görsel yorumlama şu anda desteklenmiyor.", 0

    async def interpret_image_stream(self, chat_id, fileobj, prompt=None):
        yield "🖼️ Görsel yorumlama şu anda desteklenmiyor.", "0"

    def reset_chat_history(self, chat_id, content=''):
        if not content:
            content = self.config.get('assistant_prompt', 'You are a helpful assistant.')
        self.conversations[chat_id] = [{"role": "system", "content": content}]
        self.conversations_vision[chat_id] = False

    def __max_age_reached(self, chat_id) -> bool:
        if chat_id not in self.last_updated:
            return False
        return self.last_updated[chat_id] < datetime.datetime.now() - datetime.timedelta(
            minutes=self.config.get('max_conversation_age_minutes', 180))

    def __add_function_call_to_history(self, chat_id, function_name, content):
        self.conversations[chat_id].append({"role": "function", "name": function_name, "content": content})

    def __add_to_history(self, chat_id, role, content):
        self.conversations[chat_id].append({"role": role, "content": content})

    async def __summarise(self, conversation) -> str:
        resp = await self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Summarize this conversation briefly."},
                {"role": "user", "content": str(conversation)}
            ],
            temperature=0.4
        )
        return resp.choices[0].message.content

    def __max_model_tokens(self):
        return 4096

    def __count_tokens(self, messages) -> int:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.get_encoding("o200k_base")
        return sum(len(enc.encode(str(m))) for m in messages)
