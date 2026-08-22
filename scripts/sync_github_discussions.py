#!/usr/bin/env python3

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ID = "MDEwOlJlcG9zaXRvcnkzMzczOTQxNzM="
CATEGORY_ID = "DIC_kwDOFBw5_c4DCa1e"
GRAPHQL_URL = "https://api.github.com/graphql"


def discussion_title(post):
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})-(.+)\.md", post.name)
    if not match:
        return None
    year, month, day, slug = match.groups()
    return f"{year}/{month}/{day}/{slug}"


def graphql(token, query, variables):
    request = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "jaimeyzzz-blog-discussion-sync",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode(errors="replace")) from error
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], ensure_ascii=False))
    return payload["data"]


def existing_titles(token):
    query = """
    query($repository: ID!, $category: ID!, $cursor: String) {
      node(id: $repository) {
        ... on Repository {
          discussions(first: 100, after: $cursor, categoryId: $category) {
            nodes { title }
            pageInfo { hasNextPage endCursor }
          }
        }
      }
    }
    """
    titles = set()
    cursor = None
    while True:
        data = graphql(
            token,
            query,
            {"repository": REPOSITORY_ID, "category": CATEGORY_ID, "cursor": cursor},
        )
        discussions = data["node"]["discussions"]
        titles.update(item["title"] for item in discussions["nodes"])
        if not discussions["pageInfo"]["hasNextPage"]:
            return titles
        cursor = discussions["pageInfo"]["endCursor"]


def create_discussion(token, title):
    mutation = """
    mutation($repository: ID!, $category: ID!, $title: String!, $body: String!) {
      createDiscussion(input: {
        repositoryId: $repository,
        categoryId: $category,
        title: $title,
        body: $body
      }) {
        discussion { url }
      }
    }
    """
    source_url = f"https://www.jaimeyzzz.com/{title}.html"
    data = graphql(
        token,
        mutation,
        {
            "repository": REPOSITORY_ID,
            "category": CATEGORY_ID,
            "title": title,
            "body": f"Comments for [{source_url}]({source_url}).",
        },
    )
    return data["createDiscussion"]["discussion"]["url"]


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required.")

    desired = {
        title
        for post in (ROOT / "_posts").glob("*.md")
        if (title := discussion_title(post))
    }
    existing = existing_titles(token)
    missing = sorted(desired - existing)
    if not missing:
        print("All post discussions already exist.")
        return

    for title in missing:
        print(f"Created {title}: {create_discussion(token, title)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
