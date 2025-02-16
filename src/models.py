from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict


# Pydantic model for API/validation
class CategoryModel(BaseModel):
    id: int
    name: str
    active: bool
    rank: float
    slug: str
    scraped_at: datetime = datetime.now(tz=timezone.utc)

    model_config = ConfigDict(
        json_encoders={datetime: lambda dt: dt.isoformat()}, from_attributes=True
    )


class SubstackModel(BaseModel):
    # Core identifiers
    id: int
    author_id: int
    subdomain: str
    base_url: str
    custom_domain: str | None
    category_id: int | None

    # Publication info
    name: str
    author_name: str | None
    author_bio: str | None
    author_photo_url: str | None
    twitter_screen_name: str | None
    type: str
    language: str
    scraped_at: datetime = datetime.now(tz=timezone.utc)

    # Popularity metrics
    freeSubscriberCount: int | None
    rankingDetail: str | None
    rankingDetailFreeIncluded: str | None
    tier: int | None

    # Content indicators
    has_posts: bool
    podcast_enabled: bool
    last_chat_post_at: datetime | None

    # Visual assets
    logo_url: str | None
    cover_photo_url: str | None

    model_config = ConfigDict(
        json_encoders={datetime: lambda dt: dt.isoformat()}, from_attributes=True
    )


class PostModel(BaseModel):
    # Core identifiers
    id: int
    publication_id: int
    title: str
    subtitle: str | None
    slug: str
    type: str
    audience: str
    canonical_url: str
    section_id: int | None

    # Content metadata
    description: str | None
    wordcount: int | None
    cover_image: str | None
    cover_image_is_square: bool | None
    cover_image_is_explicit: bool | None
    search_engine_title: str | None
    search_engine_description: str | None
    social_title: str | None
    body_html: str | None
    truncated_body_text: str | None

    # Timestamps
    post_date: datetime
    scraped_at: datetime = datetime.now(tz=timezone.utc)
    updated_at: datetime | None = None

    # Engagement metrics
    reaction_count: int | None
    comment_count: int | None
    child_comment_count: int | None
    likes: int | None
    restacks: int | None = None

    # Navigation
    previous_post_slug: str | None = None
    next_post_slug: str | None = None

    # repost
    restacked_post_id: int | None = None
    restacked_post_slug: str | None = None
    restacked_pub_name: str | None = None
    restacked_pub_logo_url: str | None = None

    # Settings
    write_comment_permissions: str | None
    should_send_free_preview: bool
    free_unlock_required: bool
    default_comment_sort: str | None
    audience_before_archived: str | None = None
    editor_v2: bool | None
    exempt_from_archive_paywall: bool | None = None
    show_guest_bios: bool | None = None
    teaser_post_eligible: bool | None
    is_metered: bool | None = None
    is_geoblocked: bool | None
    hidden: bool | None = None
    hasCashtag: bool | None

    # Media fields
    has_voiceover: bool | None

    # Additional metadata
    postTags: str | None = None

    model_config = ConfigDict(
        json_encoders={datetime: lambda dt: dt.isoformat()}, from_attributes=True
    )
