import os
import uuid
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from instagrapi import Client


app = FastAPI(title="SocialPilot Instagram API")

API_KEY = os.environ["API_KEY"]
IG_USERNAME = os.environ["IG_USERNAME"]
IG_PASSWORD = os.environ["IG_PASSWORD"]

SESSION_FILE = "/tmp/instagram_session.json"


class PublishRequest(BaseModel):
    caption: str
    openaiFileIdRefs: list[Any]
    topic: str | None = None
    primary_keyword: str | None = None
    hashtags: list[str] | None = None
    account_id: str | None = None


def check_auth(authorization: str | None):
    expected = f"Bearer {API_KEY}"

    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def get_instagram_client():
    cl = Client()

    # Try to reuse the session while this Render instance exists
    if os.path.exists(SESSION_FILE):
        try:
            cl.load_settings(SESSION_FILE)
        except Exception:
            pass

    cl.login(IG_USERNAME, IG_PASSWORD)

    # Save refreshed session
    cl.dump_settings(SESSION_FILE)

    return cl


def download_openai_image(file_ref):
    """
    GPT Actions describes openaiFileIdRefs as strings in the OpenAPI schema,
    but at runtime OpenAI sends objects containing download_link,
    mime_type, id and name.
    """

    if not isinstance(file_ref, dict):
        raise HTTPException(
            status_code=400,
            detail="Invalid OpenAI file reference"
        )

    download_url = file_ref.get("download_link")

    if not download_url:
        raise HTTPException(
            status_code=400,
            detail="Image download link missing"
        )

    response = requests.get(download_url, timeout=30)
    response.raise_for_status()

    original_path = Path(f"/tmp/{uuid.uuid4()}")
    original_path.write_bytes(response.content)

    # Convert whatever ChatGPT gives us into JPEG for Instagram
    output_path = Path(f"/tmp/{uuid.uuid4()}.jpg")

    try:
        image = Image.open(original_path)

        # JPEG cannot contain transparency
        if image.mode != "RGB":
            image = image.convert("RGB")

        image.save(
            output_path,
            format="JPEG",
            quality=95
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {str(e)}"
        )

    finally:
        try:
            original_path.unlink()
        except Exception:
            pass

    return output_path


@app.get("/")
def root():
    return {
        "service": "SocialPilot Instagram API",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "success": True,
        "status": "healthy"
    }


@app.post("/instagram/publish")
def publish_instagram(
    body: PublishRequest,
    authorization: str | None = Header(default=None)
):
    check_auth(authorization)

    if len(body.openaiFileIdRefs) != 1:
        raise HTTPException(
            status_code=400,
            detail="Exactly one image is required"
        )

    image_path = download_openai_image(
        body.openaiFileIdRefs[0]
    )

    try:
        cl = get_instagram_client()

        media = cl.photo_upload(
            image_path,
            caption=body.caption
        )

        return {
            "success": True,
            "media_id": str(media.pk),
            "media_code": media.code,
            "post_url": f"https://www.instagram.com/p/{media.code}/",
            "sheet_logged": False
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Instagram publishing failed: {str(e)}"
        )

    finally:
        try:
            image_path.unlink()
        except Exception:
            pass
