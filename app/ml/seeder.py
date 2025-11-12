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
        Embed(text="buy eggs", domain=Domain.PANTRY, intent=Intent.ADD_ITEM_TO_PANTRY),
        Embed(text="i have bought eggs", domain=Domain.PANTRY, intent=Intent.ADD_ITEM_TO_PANTRY),
        Embed(text="i have bought everything on the shopping list", domain=Domain.PANTRY, intent=Intent.ADD_ITEM_TO_PANTRY),
        Embed(text="i ate the oranges", domain=Domain.PANTRY, intent=Intent.REMOVE_ITEM_FROM_PANTRY),
        Embed(text="used all cumin", domain=Domain.PANTRY, intent=Intent.REMOVE_ITEM_FROM_PANTRY),
        Embed(text="list pantry items", domain=Domain.PANTRY, intent=Intent.CHECK_STOCK),
        # Expenses domain
        Embed(text="check last month's spending", domain=Domain.EXPENSES, intent=Intent.CHECK_EXPENSES),
        Embed(text="how much did i spend last month?", domain=Domain.EXPENSES, intent=Intent.CHECK_EXPENSES),
        Embed(text="have i spent over budget?", domain=Domain.EXPENSES, intent=Intent.CHECK_EXPENSES),
        Embed(text="pay electricity bill", domain=Domain.EXPENSES, intent=Intent.ADD_EXPENSE),
        Embed(text="send bizum payment", domain=Domain.EXPENSES, intent=Intent.ADD_EXPENSE),
        Embed(text="monthly salary", domain=Domain.EXPENSES, intent=Intent.ADD_INCOME),
        Embed(text="receive bizum payment", domain=Domain.EXPENSES, intent=Intent.ADD_INCOME),
        # Tasks domain
        Embed(text="need to call mom", domain=Domain.TASKS, intent=Intent.CREATE_TASK),
        Embed(text="schedule dentist appointment", domain=Domain.TASKS, intent=Intent.CREATE_TASK),
        Embed(text="remind me to water the plants", domain=Domain.TASKS, intent=Intent.CREATE_TASK),
        Embed(text="i have called mom", domain=Domain.TASKS, intent=Intent.COMPLETE_TASK),
        Embed(text="dentist appointment done", domain=Domain.TASKS, intent=Intent.COMPLETE_TASK),
        Embed(text="finish report", domain=Domain.TASKS, intent=Intent.COMPLETE_TASK),
    ]

    print("Seeding FAISS index with metadata...")
    texts = [item.text for item in seed_data]
    embeddings = model.encode(texts, convert_to_numpy=True)
    metadata = [{"domain": item.domain.value, "intent": item.intent.value} for item in seed_data]

    store.add(texts, embeddings, metadata)
    print(f"FAISS index total vectors: {store.index.ntotal}")
    store.cursor.execute("SELECT COUNT(*) FROM metadata")
    print(f"Metadata rows: {store.cursor.fetchone()[0]}")
    print("Seeding completed.")