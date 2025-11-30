from pydantic import BaseModel

class OrderInput(BaseModel):
    order_id: str
    customer_id: str
    # Add other fields
