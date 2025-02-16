from datetime import datetime, timezone

from scrape_substack import newsletter
from tqdm import tqdm

import src.db as db
from src.models import PostModel


def get_and_add_posts_from_substack_to_db(subdomain: str):
    posts = newsletter.get_newsletter_post_metadata(subdomain)
    for post in tqdm(posts, leave=False, desc=f"Posts from {subdomain}"):
        try:
            post_content = newsletter.get_post_contents(subdomain, post["slug"])
        except Exception as e:
            if "404" in str(e):
                print(f"404 error for post {post['slug']} in {subdomain}")
                continue
            raise e
        assert isinstance(post_content, dict)
        post_content["likes"] = post_content.get("reactions", {}).get("❤️", 0)
        tags = post_content.get("postTags", [])
        post_content["postTags"] = (
            ",".join([tag["name"] for tag in tags]) if bool(tags) else None
        )
        joint_dict = {**post, **post_content}
        db.upsert_post(PostModel(**joint_dict))
    substack = db.get_substack_by_subdomain(subdomain)
    if substack:
        substack.scraped_at = datetime.now(timezone.utc)
        db.update_substack(substack.id, substack)


if __name__ == "__main__":
    substacks = db.get_substacks(order="asc")
    for substack in tqdm(substacks, desc="Processing substacks"):
        get_and_add_posts_from_substack_to_db(substack.subdomain)
