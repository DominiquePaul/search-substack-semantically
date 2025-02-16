import os

from dotenv import load_dotenv
from supabase import create_client

from src.models import CategoryModel, PostModel, SubstackModel

# Load environment variables.
load_dotenv()

# Supabase client initialization
supabase = create_client(
    os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)

# Example of how to work with categories table using native Supabase


def get_categories() -> list[CategoryModel]:
    response = supabase.table("categories").select("*").execute()
    return [CategoryModel(**category) for category in response.data]


def get_category(id: int) -> CategoryModel:
    response = supabase.table("categories").select("*").eq("id", id).execute()
    return CategoryModel(**response.data[0])


def get_all_categories() -> list[CategoryModel]:
    response = supabase.table("categories").select("*").execute()
    return [CategoryModel(**category) for category in response.data]


def create_category(category: CategoryModel):
    return (
        supabase.table("categories").insert(category.model_dump(mode="json")).execute()
    )


def update_category(id: int, category: CategoryModel):
    return (
        supabase.table("categories")
        .update(category.model_dump(mode="json"))
        .eq("id", id)
        .execute()
    )


def delete_category(id: int):
    return supabase.table("categories").delete().eq("id", id).execute()


def upsert_substack(substack: SubstackModel):
    """
    Upserts a Substack record. If the record exists (matching subdomain),
    it will be updated, otherwise a new record will be created.
    """
    return (
        supabase.table("substacks")
        .upsert(substack.model_dump(mode="json"), on_conflict="id")
        .execute()
    )


def get_substack(subdomain: int) -> SubstackModel:
    response = (
        supabase.table("substacks").select("*").eq("subdomain", subdomain).execute()
    )
    return SubstackModel(**response.data[0])


def get_substacks(
    sort_by: str = "scraped_at", order: str = "desc", limit: int | None = None
) -> list[SubstackModel]:
    query = supabase.table("substacks").select("*")

    # Add sorting
    query = query.order(sort_by, desc=(order.lower() == "desc"))

    # Add limit if specified
    if limit:
        query = query.limit(limit)

    response = query.execute()
    return [SubstackModel(**substack) for substack in response.data]


def get_substack_by_subdomain(subdomain: str) -> SubstackModel | None:
    response = (
        supabase.table("substacks").select("*").eq("subdomain", subdomain).execute()
    )
    return SubstackModel(**response.data[0]) if response.data else None


def update_substack(id: int, substack: SubstackModel):
    return (
        supabase.table("substacks")
        .update(substack.model_dump(mode="json"))
        .eq("id", id)
        .execute()
    )


def delete_substack(id: int):
    return supabase.table("substacks").delete().eq("id", id).execute()


def upsert_post(post: PostModel):
    """
    Upserts a Post record. If the record exists (matching id),
    it will be updated, otherwise a new record will be created.
    """
    return (
        supabase.table("posts")
        .upsert(post.model_dump(mode="json"), on_conflict="id")
        .execute()
    )


def get_post(id: int) -> PostModel | None:
    """
    Retrieves a post by its ID.
    Returns None if no post is found.
    """
    response = supabase.table("posts").select("*").eq("id", id).execute()
    return PostModel(**response.data[0]) if response.data else None


def get_posts(
    publication_id: int | None = None,
    sort_by: str = "post_date",
    order: str = "desc",
    limit: int | None = None,
) -> list[PostModel]:
    """
    Retrieves posts with optional filtering by publication_id,
    sorting, and limit parameters.
    """
    query = supabase.table("posts").select("*")

    # Add publication filter if specified
    if publication_id:
        query = query.eq("publication_id", publication_id)

    # Add sorting
    query = query.order(sort_by, desc=(order.lower() == "desc"))

    # Add limit if specified
    if limit:
        query = query.limit(limit)

    response = query.execute()
    return [PostModel(**post) for post in response.data]
