from sentence_transformers import SentenceTransformer
from app.entities.enums import Intent, Domain
from app.ml.embed import Embed

def seed_faiss(store):
    """Seed the FAISS index with sample data."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    seed_data = [
        Embed(text="add milk to shopping list", domain=Domain.SHOPPING, intent=Intent.ADD_ITEM_TO_LIST),
        Embed(text="add detergent to shopping list", domain=Domain.SHOPPING, intent=Intent.ADD_ITEM_TO_LIST),
        Embed(text="i need to buy milk", domain=Domain.SHOPPING, intent=Intent.ADD_ITEM_TO_LIST),
        Embed(text="i want to buy tomatoes", domain=Domain.SHOPPING, intent=Intent.ADD_ITEM_TO_LIST),
        # Pantry domain
        {"text": "buy eggs", "domain": "pantry", "intent": "add_item_to_pantry"},
        {"text": "i have bought eggs", "domain": "pantry", "intent": "add_item_to_pantry"},
        {"text": "i have bought everything on the shopping list", "domain": "pantry", "intent": "add_item_to_pantry"},
        {"text": "i ate the oranges", "domain": "pantry", "intent": "remove_item_from_pantry"},
        {"text": "used all cumin", "domain": "pantry", "intent": "remove_item_from_pantry"},
        {"text": "list pantry items", "domain": "pantry", "intent": "check_stock"},
        # Expenses domain
        {"text": "check last month's spending", "domain": "expenses", "intent": "check_expenses"},
        {"text": "how much did i spend last month?", "domain": "expenses", "intent": "check_expenses"},
        {"text": "have i spent over budget?", "domain": "expenses", "intent": "check_expenses"},
        {"text": "pay electricity bill", "domain": "expenses", "intent": "add_expense"},
        {"text": "send bizum payment", "domain": "expenses", "intent": "add_expense"},
        {"text": "monthly salary", "domain": "expenses", "intent": "add_income"},
        {"text": "receive bizum payment", "domain": "expenses", "intent": "add_income"},
        # Tasks domain
        {"text": "need to call mom", "domain": "tasks", "intent": "create_task"},
        {"text": "schedule dentist appointment", "domain": "tasks", "intent": "create_task"},
        {"text": "remind me to water the plants", "domain": "tasks", "intent": "create_task"},
        {"text": "i have called mom", "domain": "tasks", "intent": "complete_task"},
        {"text": "dentist appointment done", "domain": "tasks", "intent": "complete_task"},
        {"text": "finish report", "domain": "tasks", "intent": "complete_task"},
    ]

    # --- 3. Embed texts ---
    texts = [item.text for item in seed_data]
    embeddings = model.encode(texts, convert_to_numpy=True)

    # --- 4. Add to VectorStore ---
    store.add(texts, embeddings)
    print("FAISS index seeded with metadata")
