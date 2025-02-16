from datetime import datetime, timezone

from scrape_substack import newsletter
from tqdm import tqdm

import src.db as db
from src.models import SubstackModel


def add_substacks_from_category_to_db(category_id: int):
    # Load existing newsletters if the file exists
    substacks = newsletter.get_newsletters_in_category(category_id)
    for substack in substacks:
        if substack["freeSubscriberCount"] is not None:
            substack["freeSubscriberCount"] = float(
                substack["freeSubscriberCount"].replace(",", "")
            )
        if substack.get("twitter_screen_name") is None:
            substack["twitter_screen_name"] = None
        if substack.get("author_name") is None:
            substack["author_name"] = None
        db.upsert_substack(SubstackModel(**substack, category_id=category_id))
    category = db.get_category(category_id)
    category.scraped_at = datetime.now(timezone.utc)
    db.update_category(category_id, category)


def add_all_categories():
    all_categories = db.get_all_categories()
    all_categories.sort(
        key=lambda x: x.scraped_at or datetime.min.replace(tzinfo=timezone.utc)
    )
    categories = db.get_categories()
    for category in tqdm(categories):
        add_substacks_from_category_to_db(category.id)


if __name__ == "__main__":
    add_all_categories()
