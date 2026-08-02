# WeChat draft sync

Install dependencies:

```bash
python3 -m pip install -r scripts/requirements.txt
```

Generate a local preview for the latest published post:

```bash
python3 scripts/wechat_sync.py
```

Create a WeChat Official Account draft:

```bash
python3 scripts/wechat_sync.py --submit
```

Use `--post _posts/YYYY-MM-DD-slug.md` to select a specific article. Credentials are read from the ignored local `.secrets/wechat.env`:

```dotenv
WECHAT_APP_ID=your_app_id
WECHAT_APP_SECRET=your_app_secret
```

Keep this file local and set its permission to `600`. The script builds Jekyll, renders formulas and inline SVG figures as images, generates a cover from the site slogan and article title, uploads the assets, and writes the converted article to the WeChat draft box. Pass `--cover PATH` to override the generated cover.

On submission, the script looks up drafts by exact article title: an existing draft is updated, while a new title creates a new draft. Use `--update-media-id MEDIA_ID --submit` only when you need to override automatic matching.
