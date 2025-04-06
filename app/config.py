from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ALLOW_ORIGINS: str = '*'
    OPENAI_API_KEY: str
    MODEL: str = 'gemini-1.5-flash-latest'  # ✅ Annotated
    EMBEDDING_MODEL: str = 'models/gemini-embedding-exp-03-07'  # ✅ Annotated
    EMBEDDING_DIMENSIONS: int = 768
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6380
    DOCS_DIR: str = 'data/docs'
    EXPORT_DIR: str = 'data'
    VECTOR_SEARCH_TOP_K: int = 10

    model_config = SettingsConfigDict(env_file='.env')

settings = Settings()