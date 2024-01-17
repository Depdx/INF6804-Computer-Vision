from src.database import Database

db = Database(directory="./data/database")

images = db.get_images()
print(next(images))
