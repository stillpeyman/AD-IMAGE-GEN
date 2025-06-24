from app import app
from datamanager.models import db


with app.app_context():
    db.create_all()
    print("Database and tables created.")
