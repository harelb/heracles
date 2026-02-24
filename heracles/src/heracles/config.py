from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class HeraclesSettings(BaseSettings):
    neo4j_uri: str = Field("neo4j://localhost:7687", validation_alias="HERACLES_NEO4J_URI")
    neo4j_username: str = Field("neo4j", validation_alias="HERACLES_NEO4J_USERNAME")
    neo4j_password: str = Field("password", validation_alias="HERACLES_NEO4J_PASSWORD")
    
    # Optional OpenAI key if using LLM features
    openai_api_key: Optional[str] = Field(None, validation_alias="HERACLES_OPENAI_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

def get_settings() -> HeraclesSettings:
    return HeraclesSettings()
