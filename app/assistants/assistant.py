import asyncio
from time import time
from app.db import get_chat_messages, add_chat_messages
from app.assistants.tools import QueryKnowledgeBaseTool
from app.assistants.prompts import MAIN_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from app.utils.sse_stream import SSEStream
import google.generativeai as genai
from app.config import settings

MODEL: str = settings.MODEL

class GeminiRAGAssistant:
    def __init__(self, chat_id, rdb, history_size=4, max_tool_calls=3):
        self.chat_id = chat_id
        self.rdb = rdb
        self.sse_stream = None
        self.main_system_prompt = MAIN_SYSTEM_PROMPT
        self.rag_system_prompt = RAG_SYSTEM_PROMPT
        self.history_size = history_size
        self.max_tool_calls = max_tool_calls
        self.model = genai.GenerativeModel(
            MODEL,
            safety_settings=[
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            ]
        )

    async def _generate_chat_response(self, chat_messages, tools=None, use_rag_prompt=False):
        try:
            injected_prompt = self.rag_system_prompt if use_rag_prompt else self.main_system_prompt
            injected_message = {'role': 'user', 'parts': injected_prompt}
            full_messages = [injected_message] + chat_messages

            response = self.model.generate_content(
                full_messages,
                stream=False,
                tools=tools,
            )

            return {"role": "assistant", "content": response.text}

        except Exception as e:
            print(f"Gemini API Error: {e}")
            if self.sse_stream:
                await self.sse_stream.send("An error occurred while generating a response.")
            return {"role": "assistant", "content": "An error occurred while generating a response."}

    async def _handle_tool_calls(self, tool_calls, chat_messages):
        for tool_call in tool_calls[:self.max_tool_calls]:
            if tool_call.name == "QueryKnowledgeBaseTool":
                kb_tool = tool_call.args
                kb_result = await QueryKnowledgeBaseTool.from_gemini_args(kb_tool, self.rdb)
                chat_messages.append({'role': 'tool', 'tool_call_id': tool_call.name, 'parts': kb_result})

        return await self._generate_chat_response(chat_messages, tools=None, use_rag_prompt=True)

    async def _run_conversation_step(self, message):
        user_db_message = {'role': 'user', 'parts': message, 'created': int(time())}
        chat_messages = await get_chat_messages(self.rdb, self.chat_id, last_n=self.history_size)
        chat_messages.append({'role': 'user', 'parts': message})

        tools = []
        if hasattr(QueryKnowledgeBaseTool, "to_gemini_tool"):
            tools = [QueryKnowledgeBaseTool.to_gemini_tool()]

        assistant_message = await self._generate_chat_response(chat_messages, tools=tools)

        tool_calls = []
        if (
            assistant_message
            and "parts" in assistant_message
            and isinstance(assistant_message["parts"], list)
            and len(assistant_message["parts"]) > 0
            and hasattr(assistant_message["parts"][0], "function_call")
            and assistant_message["parts"][0].function_call is not None
        ):
            tool_calls = [assistant_message["parts"][0].function_call]

        if tool_calls:
            assistant_message = await self._handle_tool_calls(tool_calls, chat_messages)

        assistant_db_message = {
            'role': 'assistant',
            'content': assistant_message["content"],
            'tool_calls': tool_calls,
            'created': int(time())
        }

        await add_chat_messages(self.rdb, self.chat_id, [user_db_message, assistant_db_message])
        if self.sse_stream:
            await self.sse_stream.send(assistant_db_message["content"])

    async def _handle_conversation_task(self, message):
        try:
            await self._run_conversation_step(message)
        except Exception as e:
            print(f'Error: {str(e)}')
            if self.sse_stream:
                await self.sse_stream.send(f"Error: {str(e)}")
        finally:
            if self.sse_stream:
                await self.sse_stream.close()

    def run(self, message):
        self.sse_stream = SSEStream()
        asyncio.create_task(self._handle_conversation_task(message))
        return self.sse_stream