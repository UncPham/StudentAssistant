import uuid;
from pydantic import BaseModel, Field

class VectorStore(BaseModel):
    id: str = Field(default_factory=uuid.uuid4, alias="_id")
    vector: List[float]
    text: str
    metadata: dict = {}