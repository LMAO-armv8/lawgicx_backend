import asyncio
from rich.console import Console
from app.db import get_redis2
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
        self.main_system_prompt = MAIN_SYSTEM_PROMPT
        self.rag_system_prompt = RAG_SYSTEM_PROMPT
        self.history_size = history_size
        self.max_tool_calls = max_tool_calls
        self.log_tool_calls = log_tool_calls
        self.log_tool_results = log_tool_results
        self.model = genai.GenerativeModel(
            model_name,
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

    async def _generate_chat_response(self, system_prompt, chat_messages, tools=None):
        system_prompt_message = {'role': 'user', 'parts': system_prompt}
        messages = [system_prompt_message] + chat_messages

        try:
            response = self.model.generate_content(
                messages,
                stream=False,
                tools=tools,
            )

            self.console.print(f"Raw Gemini Response: {response}", style="yellow") # Add this line

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
                
            print(tools)

            assistant_message = await self._generate_chat_response(
                system_prompt=self.main_system_prompt,
                chat_messages=chat_messages,
                tools=tools,
            )

            function_call = None
            if (
                assistant_message
                and isinstance(assistant_message, dict)
                and "parts" in assistant_message
                and isinstance(assistant_message["parts"], list)
            ):
                for part in assistant_message["parts"]:
                    if hasattr(part, "function_call") and part.function_call is not None:
                        function_call = part.function_call
                        break

            if function_call:
                if function_call.name == "QueryKnowledgeBaseTool":
                    kb_tool = function_call.args
                    kb_result = await QueryKnowledgeBaseTool.from_gemini_args(kb_tool, self.rdb)

                    if self.log_tool_results:
                        self.console.print(f'TOOL RESULT:\n{kb_result}', style='magenta', end='\n\n')

                    chat_messages.append(
                        {'role': 'tool', 'tool_call_id': function_call.name, 'parts': kb_result}
                    )

                    assistant_message = await self._generate_chat_response(
                        system_prompt=self.rag_system_prompt,
                        chat_messages=chat_messages,
                    )

            self.chat_history.extend([user_message, {'role': 'assistant', 'parts': assistant_message["content"]}])

async def run_local_gemini_assistant():
    async with get_redis2() as rdb:
        await LocalGeminiRAGAssistant(rdb).run()
def main():
    asyncio.run(run_local_gemini_assistant())

if __name__ == '__main__':
    main()