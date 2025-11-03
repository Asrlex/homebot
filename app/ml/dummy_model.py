class DummyModel:
    """
    Placeholder ML model that just echoes the input.
    Replace with embedding + intent logic later.
    """
    def predict(self, text: str) -> str:
        if "egg" in text.lower():
            return "Intent: add_item"
        elif "expense" in text.lower():
            return "Intent: check_expenses"
        else:
            return "Intent: unknown"
