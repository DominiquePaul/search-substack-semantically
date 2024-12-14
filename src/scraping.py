import pickle
import pandas as pd
from substack_api import newsletter, user
from tqdm import tqdm
import time
from typing import Dict, List

pd.set_option("display.max_columns", None)

# TODO: Cover all categogies in the future
# newsletter.list_all_categories()

CATEGORY_ID = 4


###########################################
### Scrape all newsletters in category
###########################################

relevant_columns_newsletters = [
    # Core identifiers
    "id",
    "author_id",
    "subdomain",
    "base_url",
    "custom_domain",
    # Publication info
    "name",
    "author_name",
    "author_bio",
    "author_photo_url",
    "twitter_screen_name",
    "type",
    "language",
    "sections",
    "created_at",
    # Popularity metrics
    "freeSubscriberCount",
    "rankingDetail",
    "rankingDetailFreeIncluded",
    "tier",
    # Content indicators
    "has_posts",
    "podcast_enabled",
    "last_chat_post_at",
    # Visual assets
    "logo_url",
    "cover_photo_url",
]

# Load existing newsletters if the file exists
try:
    # Load raw data
    with open("data/newsletters_raw.pkl", "rb") as f:
        newsletters_df = pickle.load(f)
    # Load filtered data
    with open("data/newsletters_filtered.pkl", "rb") as f:
        newsletters_df_filtered = pickle.load(f)
    print(f"Loaded {len(newsletters_df)} existing newsletters")
except FileNotFoundError:
    newsletters = newsletter.get_newsletters_in_category(4, start_page=0, end_page=None)
    print(f"Found {len(newsletters)} newsletters in category {CATEGORY_ID}")
    newsletters_df = pd.DataFrame(newsletters)

    # Filter columns
    newsletters_df_filtered = newsletters_df[relevant_columns_newsletters]

    # Save raw newsletters
    with open("data/newsletters_raw.pkl", "wb") as f:
        pickle.dump(newsletters_df, f)

    # Save filtered newsletters
    with open("data/newsletters_filtered.pkl", "wb") as f:
        pickle.dump(newsletters_df_filtered, f)


###########################################
### Get all posts for each newsletter
###########################################

num_newsletters = len(newsletters_df_filtered["subdomain"].unique())
print(f"Found {num_newsletters} unique newsletter subdomains")

# Load existing posts if the file exists
try:
    with open("data/post_metadata_by_newsletter.pkl", "rb") as f:
        post_metadata_by_newsletter = pickle.load(f)
    with open("data/post_contents_by_newsletter.pkl", "rb") as f:
        post_contents_by_newsletter = pickle.load(f)
except FileNotFoundError:
    post_metadata_by_newsletter = {}
    post_contents_by_newsletter = {}

# Only process newsletters not already in the dictionary
remaining_subdomains = [
    s
    for s in newsletters_df_filtered["subdomain"].unique()
    if s not in post_metadata_by_newsletter or s not in post_contents_by_newsletter
]

print(f"Processing {len(remaining_subdomains)} remaining newsletters...")


def process_newsletter(
    subdomain: str, retry_count: int = 3, delay: float = 1.0
) -> Dict:
    failed_slugs = []
    for attempt in range(retry_count):
        try:
            posts_metadata = newsletter.get_newsletter_post_metadata(
                subdomain, start_offset=0, end_offset=1000
            )
            time.sleep(delay)  # Rate limiting

            if not isinstance(posts_metadata, list):  # Basic validation
                raise ValueError(f"Unexpected data format for {subdomain}")

            posts_metadata_df = pd.DataFrame(posts_metadata)
            post_contents = {}

            for slug in tqdm(posts_metadata_df["slug"], leave=False):
                try:
                    content = newsletter.get_post_contents(
                        subdomain, slug, html_only=False
                    )
                    time.sleep(delay)  # Rate limiting
                    post_contents[slug] = content
                except Exception as e:
                    failed_slugs.append(slug)
                    print(f"Failed to get content for {slug}: {str(e)}")

            return {
                "metadata": posts_metadata_df,
                "contents": post_contents,
                "failed_slugs": failed_slugs,
            }

        except Exception as e:
            if attempt == retry_count - 1:
                raise
            print(f"Attempt {attempt + 1} failed for {subdomain}: {str(e)}")
            time.sleep(delay * (attempt + 1))  # Exponential backoff

    return None


for subdomain in tqdm(
    remaining_subdomains,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
):
    tqdm.postfix = f"Current: {subdomain}"
    try:
        result = process_newsletter(subdomain)
        if result:
            post_metadata_by_newsletter[subdomain] = result["metadata"]
            post_contents_by_newsletter[subdomain] = result["contents"]

            # Save progress after each newsletter
            with open("data/post_metadata_by_newsletter.pkl", "wb") as f:
                pickle.dump(post_metadata_by_newsletter, f)
            with open("data/post_contents_by_newsletter.pkl", "wb") as f:
                pickle.dump(post_contents_by_newsletter, f)

    except Exception as e:
        print(f"Error processing {subdomain}: {str(e)}")
        continue
