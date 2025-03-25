from fastapi import Depends

def get_db():
    """Mock database dependency"""
    db = {"message": "Database Connection"}
    return db
