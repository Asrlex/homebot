from pydantic import BaseModel
from app.entities.enums import Domain, Intent

class Embed(BaseModel):
    text: str
    domain: Domain
    intent: Intent