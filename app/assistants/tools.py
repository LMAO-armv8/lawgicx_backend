from pydantic import BaseModel, Field
from app.db import search_vector_db
from app.openai import get_embedding  # Adapt if using Gemini embeddings

class QueryKnowledgeBaseTool(BaseModel):
    """Query the knowledge base to answer user questions about new technology trends, their applications and broader impacts."""
    query_input: str = Field(description='The natural language query input string. The query input should be clear and standalone.')

    async def __call__(self, rdb):
        query_vector = await get_embedding(self.query_input)  # Adapt for Gemini embeddings if needed
        print(query_vector)
        chunks = await search_vector_db(rdb, query_vector)
        formatted_sources = [f"SOURCE: {c['doc_name']}\n\"\"\"\n{c['text']}\n\"\"\"" for c in chunks]
        return f"\n\n---\n\n".join(formatted_sources) + f"\n\n---"

    @staticmethod
    def to_gemini_tool():
        """Converts the tool to Gemini's function calling format (snake_case + lowercase types)."""
        return {
            "function_declarations": [
                {
                    "name": "QueryKnowledgeBaseTool",
                    "description": (
                        "Query the knowledge base to answer user questions about new technology trends, "
                        "their applications and broader impacts."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_input": {
                                "type": "string",
                                "description": (
                                    "The natural language query input string. "
                                    "The query input should be clear and standalone."
                                )
                            }
                        },
                        "required": ["query_input"]
                    }
                }
            ]
        }

    @staticmethod
    async def from_gemini_args(args, rdb):
        """Converts Gemini's function call arguments to the tool's expected input."""
        tool = QueryKnowledgeBaseTool(query_input=args["query_input"])
        return await tool(rdb)
