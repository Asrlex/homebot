# Homebot

Homebot is a personal assistant bot designed to help you manage tasks, set reminders, and provide information through a conversational interface. Built with Python, Homebot leverages natural language processing to understand and respond to user commands effectively.

## Features

- FAISS vector store for efficient data retrieval
  - Pre-seed FAISS index for quick startup
- Prometheus integration for monitoring and metrics
- Modular architecture for easy customization and extension

## Installation

1. Clone the repository:
```bash
  git clone <repository-url>
  cd homebot
```
2. Install the dependencies:
```bash
  pip install -r requirements.txt
```
3. Run the application:
```bash
  uvicorn app.main:app --reload
```

## Sample Usage

1. Access the API documentation at `http://localhost:8000/redoc` to explore available endpoints.
2. Use the `/embed` endpoint to add texts to the vector store.
```bash
  curl -X POST "http://localhost:8000/api/embed" -H "Content-Type: application/json" -d '[{"text": "add bread to shopping list", "domain": "shopping", "intent": "add_item_to_list"}]'
```
3. Use the `/search` endpoint to retrieve relevant information based on a query.
```bash
  curl -X GET "http://localhost:8000/api/search?query=buy%20some%20milk&top_k=3"
  curl -X GET "http://localhost:8000/api/search?query=list%20pantry%20stock&top_k=3"
```
