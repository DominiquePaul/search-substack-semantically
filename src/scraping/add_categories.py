from scrape_substack import newsletter

from src.db import create_category, get_categories, update_category
from src.models import CategoryModel


def add_categories_to_db():
    """Add or update categories in the database"""
    db_categories = get_categories()
    categories = newsletter.list_all_categories()

    for category_data in categories:
        if category_data["id"] in [category.id for category in db_categories]:
            # Update existing category
            category = CategoryModel(**category_data)
            update_category(category_data["id"], category)
        else:
            # Create new category
            category = CategoryModel(**category_data)
            create_category(category)


if __name__ == "__main__":
    add_categories_to_db()
