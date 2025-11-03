from enum import Enum

class Domain(str, Enum):
    SHOPPING = "shopping"
    PANTRY = "pantry"
    EXPENSES = "expenses"
    TASKS = "tasks"

class Intent(str, Enum):
    ADD_ITEM_TO_LIST = "add_item_to_list"
    ADD_ITEM_TO_PANTRY = "add_item_to_pantry"
    REMOVE_ITEM_FROM_PANTRY = "remove_item_from_pantry"
    CHECK_STOCK = "check_stock"
    CHECK_EXPENSES = "check_expenses"
    ADD_EXPENSE = "add_expense"
    ADD_INCOME = "add_income"
    CREATE_TASK = "create_task"
    COMPLETE_TASK = "complete_task"
