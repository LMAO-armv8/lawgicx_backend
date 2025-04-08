import asyncio
from rich.console import Console
from app.db import get_redis
import google.generativeai as genai
from app.config import settings

from app.assistants.tools import QueryKnowledgeBaseTool
from app.assistants.prompts import MAIN_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT

MODEL: str = settings.MODEL

class LocalGeminiRAGAssistant:
    def __init__(self, rdb, history_size=4, max_tool_calls=3, log_tool_calls=True, log_tool_results=False, model_name=MODEL):
        self.console = Console()
        self.rdb = rdb
        self.chat_history = []
        self.main_system_message = {'role': 'system', 'parts': MAIN_SYSTEM_PROMPT}
        self.rag_system_message = {'role': 'system', 'parts': RAG_SYSTEM_PROMPT}
        self.history_size = history_size
        self.max_tool_calls = max_tool_calls
        self.log_tool_calls = log_tool_calls
        self.log_tool_results = log_tool_results
        self.model = genai.GenerativeModel(model_name)

    async def _generate_chat_response(self, system_message, chat_messages, tools=None):
        messages = [system_message] + chat_messages

        try:
            response = self.model.generate_content(
                chat_messages,
                stream=False,
                tools=tools,
            )

            if response.text:
                self.console.print(response.text, style='cyan', end='')

            assistant_message = {"role": "assistant", "content": response.text}
            self.console.print('\n')
            return assistant_message

        except Exception as e:
            self.console.print(f"Gemini API Error: {e}", style="red")
            return {"role": "assistant", "content": "An error occurred while generating a response."}

    async def run(self):
        self.console.print('How can I help you?\n', style='cyan')
        while True:
            chat_messages = self.chat_history[-self.history_size:]
            user_input = input()
            self.console.print()
            user_message = {'role': 'user', 'parts': user_input}
            chat_messages.append(user_message)

            tools = []
            if hasattr(QueryKnowledgeBaseTool, "to_gemini_tool"):
                tools = [QueryKnowledgeBaseTool.to_gemini_tool()]

            assistant_message = await self._generate_chat_response(
                system_message=self.main_system_message,
                chat_messages=chat_messages,
                tools=tools,
            )

            if (
                assistant_message
                and isinstance(assistant_message, dict)
                and "parts" in assistant_message
                and isinstance(assistant_message["parts"], list)
                and hasattr(assistant_message["parts"][0], "function_call")
            ):
                function_call = assistant_message["parts"][0].function_call
                if self.log_tool_calls:
                    self.console.print(f'TOOL CALL:\n{function_call}', style='red', end='\n\n')

                if function_call.name == "QueryKnowledgeBaseTool":
                    kb_tool = function_call.args
                    kb_result = await QueryKnowledgeBaseTool.from_gemini_args(kb_tool, self.rdb)

                    if self.log_tool_results:
                        self.console.print(f'TOOL RESULT:\n{kb_result}', style='magenta', end='\n\n')

                    chat_messages.append(
                        {'role': 'tool', 'tool_call_id': function_call.name, 'parts': kb_result}
                    )

                    assistant_message = await self._generate_chat_response(
                        system_message=self.rag_system_message,
                        chat_messages=chat_messages,
                    )

            self.chat_history.extend([
                user_message,
                {'role': 'assistant', 'parts': assistant_message["content"]}
            ])

async def run_local_gemini_assistant():
    async with get_redis() as rdb:
        await LocalGeminiRAGAssistant(rdb).run()

def main():
    asyncio.run(run_local_gemini_assistant())

if __name__ == '__main__':
    main()