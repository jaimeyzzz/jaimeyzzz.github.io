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

# Zhihu manual publishing

Generate a private, Zhihu-friendly rich-text copy page for the latest published post:

```bash
python3 scripts/zhihu_export.py
```

Select a specific post with `--post _posts/YYYY-MM-DD-slug.md`. The output is written under the ignored `.preview/zhihu/` directory and is available through the private preview server at `/blog-preview/zhihu/slug.html`. Copy the title first, then use the one-click rich-text body button.

Regular article images use their public blog URLs so Zhihu can import them. Formula images are embedded in the clipboard document. Always confirm every image after pasting; if Zhihu does not import a remote image, download it from the original post and insert it manually.
