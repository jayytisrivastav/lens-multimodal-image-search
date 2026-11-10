from contextlib import closing
from datetime import datetime, timezone
from time import perf_counter
import json
from hashlib import sha256
from functools import lru_cache
from math import sqrt, isfinite
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from flask import Flask, redirect, render_template_string, request, url_for
from PIL import Image, ImageOps, UnidentifiedImageError
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
STORE = ROOT / "image_store"
app = Flask(__name__, static_folder=str(IMAGES), static_url_path="/images")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
model = SentenceTransformer("jinaai/jina-clip-v2", device="cpu",
                            trust_remote_code=True, truncate_dim=1024)

# Candidate vocabulary: suggestions are relative matches, not verified labels.
TAG_LABELS = (
    "car", "truck", "bus", "motorcycle", "bicycle", "airplane", "boat",
    "butterfly", "snake", "bird", "dog", "cat", "horse", "fish",
    "flower", "tree", "forest", "lake", "river", "beach", "mountain",
    "building", "street", "bridge", "food", "fruit", "book", "computer",
    "phone", "chair", "table", "person", "shoe", "bag", "watch",
)


@lru_cache(maxsize=1)
def tag_vectors():
    return model.encode(
        ["A photo of a " + label for label in TAG_LABELS],
        normalize_embeddings=True, batch_size=8,
    ).tolist()


def suggest_tags(image_vector):
    scored = sorted(
        ((label, sum(a * b for a, b in zip(image_vector, text_vector)))
         for label, text_vector in zip(TAG_LABELS, tag_vectors())),
        key=lambda item: item[1], reverse=True,
    )
    # Return three candidates; no probability or accuracy claim.
    return [label for label, score in scored[:3]]


UI_STYLE = '<style>\n:root{--ink:#172e2b;--muted:#61736f;--green:#17644e;--line:#dce5df;--paper:#f5f7f2;--white:#fff;--mint:#e7f1e9}\n*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}\na{color:var(--green)}button,input,select{font:inherit}button,a,input,select,summary{outline-offset:4px}button{cursor:pointer;border:0}\nheader{background:white;border-bottom:1px solid var(--line)}.nav{max-width:1240px;margin:auto;display:flex;align-items:center;justify-content:space-between;padding:20px 28px;gap:20px}\n.brand{text-decoration:none;font-size:22px;font-weight:800;letter-spacing:-1px;color:var(--ink)}.brand span{background:var(--green);color:white;border-radius:10px;padding:4px 10px;margin-right:9px}\n.navlinks{display:flex;gap:8px}.navlinks a{text-decoration:none;border-radius:8px;padding:8px 16px;font-weight:600}.navlinks a.active{background:var(--mint)}\nmain{max-width:1240px;margin:auto;padding:36px 28px 60px}.eyebrow{font-size:12px;font-weight:750;letter-spacing:2px;text-transform:uppercase;color:var(--green)}h1{font-size:clamp(30px,4vw,46px);letter-spacing:-1.8px;line-height:1.15;margin:10px 0 14px}h2{font-size:22px;letter-spacing:-.5px;margin:0 0 12px}h3{font-size:17px;margin:0 0 10px}p{margin:10px 0 16px}.muted,small{color:var(--muted)}.hero{margin-bottom:30px}.hero p{max-width:690px;font-size:16px}\n.workspace{display:grid;grid-template-columns:1fr 1fr;gap:22px}.panel{background:white;border:1px solid var(--line);border-radius:16px;padding:25px;min-width:0}.step{color:var(--green);font-size:12px;font-weight:800;margin-bottom:7px;letter-spacing:1px}.field{display:block;font-size:13px;font-weight:650;margin:14px 0 6px}\ninput:not([type=checkbox]):not([type=hidden]),select{width:100%;padding:11px 12px;border:1px solid #bacbc2;border-radius:8px;background:white;color:var(--ink);min-width:0}input[type=file]{font-size:13px}\ninput::file-selector-button{border:0;border-radius:5px;background:var(--mint);color:var(--green);padding:8px 10px;margin-right:12px;font-weight:600}\n.primary{background:var(--green);color:white;border-radius:8px;padding:12px 22px;font-weight:650;display:inline-block}.primary:hover{background:#104d3b}.primary:disabled{opacity:.65}\ndetails{border-top:1px solid var(--line);margin:18px 0;padding-top:14px}summary{cursor:pointer;font-weight:650;font-size:14px}.filtergrid{display:grid;grid-template-columns:1fr 1fr;gap:0 14px}.hint{display:block;font-size:12px;margin:8px 0 12px}.upload{margin-top:22px}.upload summary{font-size:18px}.upload details{border:0;margin:0;padding:0}.uploadrow{display:grid;grid-template-columns:1fr 1fr;gap:18px}\n.check{display:flex;align-items:center;gap:8px;margin:14px 0;font-size:14px}.notice{background:var(--mint);border:1px solid #c9dfce;border-radius:10px;padding:15px 18px;margin:24px 0;overflow-wrap:anywhere}\n.resultshead{display:flex;justify-content:space-between;align-items:center;margin:28px 0 16px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(235px,1fr));gap:20px}\nfigure{margin:0;background:white;border:1px solid var(--line);border-radius:14px;overflow:hidden}.photo{display:block;width:100%;height:230px;object-fit:contain;background:#eef1eb}.caption{padding:16px}.filename{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px;margin:9px 0 14px}.rank{font-size:12px;font-weight:750;color:var(--green)}\n.score{float:right;background:var(--mint);border-radius:5px;padding:2px 7px;font-size:12px;font-weight:700}.tag{display:inline-block;border:1px solid var(--line);border-radius:5px;padding:2px 6px;margin:2px;font-size:11px}.tag.ai{border-style:dashed;color:var(--muted)}\n.taglabel{font-size:11px;color:var(--muted);margin-top:8px}.empty{text-align:center;padding:30px;color:var(--muted);border:1px dashed #c5d3c9;border-radius:12px;margin-top:24px}\n.preview{width:100%;height:150px;object-fit:contain;background:var(--paper);margin-top:10px;border-radius:8px}.preview[hidden]{display:none}.metricgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}.metric strong{display:block;font-size:34px;letter-spacing:-1px;color:var(--green)}.metric span{font-size:13px;color:var(--muted)}\n.evalcard{display:block;background:white;border:2px solid var(--line);border-radius:12px;padding:12px;overflow-wrap:anywhere;font-size:12px;cursor:pointer}.evalcard:has(input:checked){border-color:var(--green);background:var(--mint)}.evalcard img{width:100%;height:150px;object-fit:contain;margin-bottom:10px}.evalcard input{accent-color:var(--green)}.evalgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:14px;margin:20px 0}\n.tablewrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:12px;border-bottom:1px solid var(--line)}td.filenamecell{max-width:420px;overflow-wrap:anywhere}.good{color:var(--green);font-weight:650}.bad{color:#95502d}footer{margin-top:30px;font-size:12px;color:var(--muted)}.back{display:inline-block;margin-bottom:20px}\n@media(max-width:700px){.workspace,.uploadrow{grid-template-columns:1fr}.nav{padding:16px}.navlinks a{padding:6px 9px}main{padding:25px 16px}.panel{padding:19px}.metricgrid{grid-template-columns:1fr}.metric strong{font-size:28px}.brand{font-size:19px}}\n</style>\n'
UI_SCRIPT = "<script>\ndocument.querySelectorAll('[data-preview]').forEach(input=>{\nlet objectURL;\ninput.addEventListener('change',()=>{\nconst preview=document.getElementById(input.dataset.preview);\nif(objectURL)URL.revokeObjectURL(objectURL);\nconst file=input.files[0];\npreview.hidden=!file;\nif(file){objectURL=URL.createObjectURL(file);preview.src=objectURL;}\nelse preview.removeAttribute('src');\n});\n});\ndocument.querySelectorAll('form').forEach(form=>form.addEventListener('submit',()=>{\nconst button=form.querySelector('button');\nif(button){button.dataset.label=button.textContent;button.textContent='Working…';button.disabled=true;}\nform.setAttribute('aria-busy','true');\n}));\nwindow.addEventListener('pageshow',()=>document.querySelectorAll('form').forEach(form=>{\nform.removeAttribute('aria-busy');\nform.querySelectorAll('button').forEach(b=>{b.disabled=false;if(b.dataset.label)b.textContent=b.dataset.label;});\n}));\n</script>\n"

@app.context_processor
def ui_assets():
    return {"ui_style": UI_STYLE, "ui_script": UI_SCRIPT}

PAGE = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lens — Image Search</title>\n{{ ui_style|safe }}</head><body>\n<header><nav class="nav"><a class="brand" href="/"><span>L</span>Lens</a><div class="navlinks"><a class="active" href="/">Search</a><a href="/evaluate">Evaluate</a></div></nav></header>\n<main>\n<section class="hero"><div class="eyebrow">Your visual search workspace</div><h1>Find the picture you have in mind.</h1><p class="muted">Describe it, show a reference, or combine both. Search the images in your collection.</p></section>\n{% macro controls() %}\n<details><summary>Filters &amp; result settings</summary><div class="filtergrid">\n<label><span class="field">Maximum results</span><input type="number" name="limit" min="1" max="20" value="{{ options.limit }}"></label>\n<label><span class="field">Minimum similarity</span><input type="number" name="minimum" min="-1" max="1" step="0.01" value="{{ options.minimum }}"></label>\n<label><span class="field">Exact tag</span><input name="tag" value="{{ options.tag }}" placeholder="Any tag"></label>\n<label><span class="field">Tag source</span><select name="source"><option value="manual" {% if options.source == \'manual\' %}selected{% endif %}>Manual tags</option><option value="auto" {% if options.source == \'auto\' %}selected{% endif %}>AI suggestions</option></select></label>\n</div><span class="hint">Higher similarity means a closer match, not a confidence percentage. -1 includes all scores. AI tags may be incorrect.</span></details>\n{% endmacro %}\n<div class="workspace">\n<section class="panel"><div class="step">01 / DESCRIBE</div><h2>Search with words</h2><p class="muted">Try “a blue car beside a lake” or a manual tag.</p>\n<form action="/#results" method="get"><label class="field" for="textquery">What are you looking for?</label><input id="textquery" name="q" value="{{ query }}" required maxlength="500" placeholder="Describe a picture…">\n{{ controls() }}<button class="primary">Search images</button></form></section>\n<section class="panel"><div class="step">02 / SHOW</div><h2>Search with a picture</h2><p class="muted">Add an optional description to refine your search.</p>\n<form action="/search_image#results" method="post" enctype="multipart/form-data">\n<label class="field" for="reference">Reference picture</label><input id="reference" type="file" name="reference" accept="image/jpeg,image/png" required data-preview="reference-preview">\n<img class="preview" id="reference-preview" hidden alt="Selected reference picture">\n<label class="field" for="description">Optional description</label><input id="description" name="description" maxlength="500" value="{{ description|default(\'\') }}" placeholder="e.g. a car beside a lake">\n<span class="hint">Searches existing pictures. Your reference is not added to the collection.</span>\n{{ controls() }}<button class="primary">Find similar images</button></form></section></div>\n<section class="panel upload"><details><summary>Add a picture to your collection</summary>\n<form action="/upload_image" method="post" enctype="multipart/form-data">\n<div class="uploadrow"><label><span class="field">Picture · JPG or PNG</span><input type="file" name="picture" accept="image/jpeg,image/png" required data-preview="upload-preview"><img class="preview" id="upload-preview" hidden alt="Picture to upload"></label>\n<label><span class="field">Manual tags · optional</span><input name="tags" placeholder="nature, garden, butterfly"></label></div>\n<label class="check"><input type="checkbox" name="auto_tags" value="1" checked>Suggest tags automatically</label>\n<p class="hint">Three suggestions from a fixed vocabulary. Check their relevance; manual tags are kept separately.</p><button class="primary">Upload &amp; index</button></form></details></section>\n<section id="results" aria-live="polite">\n{% if message %}<div class="notice">{{ message }}</div>{% endif %}\n{% if photos %}<div class="resultshead"><h2>Your results</h2><span class="muted">{{ photos|length }} pictures</span></div><div class="grid">\n{% for photo in photos %}<figure>\n<a href="{{ url_for(\'static\', filename=photo.name) }}" target="_blank" rel="noopener"><img class="photo" src="{{ url_for(\'static\', filename=photo.name) }}" alt="{{ photo.name }}" loading="lazy"></a>\n<figcaption class="caption"><span class="rank">MATCH {{ "%02d"|format(loop.index) }}</span><span class="score">Similarity: {{ "%.3f"|format(photo.score) }}</span>\n<div class="filename" title="{{ photo.name }}">{{ photo.name }}</div>\n<div class="taglabel">MANUAL TAGS</div>{% for tag in photo.manual %}<span class="tag">{{ tag }}</span>{% else %}<small>None</small>{% endfor %}\n<div class="taglabel">AI SUGGESTIONS</div>{% for tag in photo.automatic %}<span class="tag ai">{{ tag }}</span>{% else %}<small>None</small>{% endfor %}\n</figcaption></figure>{% endfor %}</div>\n{% elif not message %}<div class="empty">Your next discovery starts with a search above.</div>{% endif %}\n</section><footer>Lens · Local image search · Click any result to view the full image.</footer>\n</main>{{ ui_script|safe }}</body></html>\n'


def search_options(values):
    try:
        limit = int(values.get("limit", "5"))
        minimum = float(values.get("minimum", "-1"))
    except (ValueError, TypeError):
        raise ValueError("Enter a whole result count and a numeric minimum similarity.")
    if not 1 <= limit <= 20 or not isfinite(minimum) or not -1 <= minimum <= 1:
        raise ValueError("Results must be 1 to 20; minimum similarity must be -1 to 1.")
    source = values.get("source", "manual")
    if source not in ("manual", "auto"):
        raise ValueError("Choose manual or AI-suggested tags.")
    return dict(limit=limit, minimum=minimum,
                tag=values.get("tag", "").strip().casefold(), source=source)


def retrieve_photos(vector, options, priority_tag=""):
    conditions = []
    if options["tag"]:
        key = "tags" if options["source"] == "manual" else "auto_tags"
        conditions.append(models.FieldCondition(
            key=key, match=models.MatchValue(value=options["tag"])))
    with closing(QdrantClient(path=str(STORE))) as client:
        if not client.collection_exists("images"):
            raise ValueError("Build your image index before searching.")
        def ranked(extra=None):
            must = conditions + ([extra] if extra else [])
            return client.query_points(
                collection_name="images", query=vector,
                query_filter=models.Filter(must=must) if must else None,
                score_threshold=options["minimum"], limit=options["limit"],
                with_payload=True,
            ).points
        candidates = []
        if priority_tag:
            candidates = ranked(models.FieldCondition(
                key="tags", match=models.MatchValue(value=priority_tag.casefold())))
        candidates += ranked()
    photos, seen = [], set()
    for point in candidates:
        payload = point.payload or {}
        if not payload.get("path"):
            continue
        name = Path(payload["path"]).name
        if name in seen:
            continue
        seen.add(name)
        photos.append(dict(name=name, score=point.score,
                           manual=payload.get("manual_tags", payload.get("tags", [])),
                           automatic=payload.get("auto_tags", [])))
        if len(photos) == options["limit"]:
            break
    return photos


@app.get("/")
def home():
    query = request.args.get("q", "").strip()
    try:
        options = search_options(request.args)
    except ValueError as error:
        return str(error), 400
    photos = []
    message = request.args.get("message", "")
    if query:
        vector = model.encode(query, prompt_name="retrieval.query",
                              normalize_embeddings=True).tolist()
        try:
            photos = retrieve_photos(vector, options, priority_tag=query)
        except ValueError as error:
            return str(error), 400
        message = ("Results: exact manual-tag matches first, then similarity."
                   if photos else "No images match these filters. Try a lower minimum or clear the tag.")
    return render_template_string(PAGE, query=query, photos=photos,
                                  message=message, options=options)


@app.errorhandler(413)
def too_large(error):
    return "Choose an image file smaller than 16 MB.", 413


@app.post("/search_image")
def search_image():
    try:
        options = search_options(request.form)
    except ValueError as error:
        return str(error), 400
    upload = request.files.get("reference")
    if not upload or not upload.filename:
        return "Choose a JPG or PNG reference picture.", 400
    try:
        with Image.open(upload, formats=["PNG", "JPEG"]) as image:
            if image.width * image.height > 24_000_000:
                return "Choose a picture under 24 megapixels.", 400
            rgb = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return "Choose a readable JPG or PNG reference picture.", 400

    description = request.form.get("description", "").strip()
    if len(description) > 500:
        return "Keep the description under 500 characters.", 400
    vector = model.encode([rgb], normalize_embeddings=True)[0].tolist()
    if description:
        text_vector = model.encode(
            description, prompt_name="retrieval.query",
            normalize_embeddings=True,
        ).tolist()
        # Equal-weight fusion in the shared image/text embedding space.
        vector = [(image_value + text_value) / 2
                  for image_value, text_value in zip(vector, text_vector)]
        norm = sqrt(sum(value * value for value in vector))
        if norm < 1e-12:
            return "Try a different description for this reference picture.", 400
        vector = [value / norm for value in vector]
    try:
        photos = retrieve_photos(vector, options)
    except ValueError as error:
        return str(error), 400
    return render_template_string(
        PAGE, query="", photos=photos, options=options, description=description,
        message=(("Image + text results for: " + description) if description
                 else "Similar images, ranked from closest match.")
                if photos else "No images match these filters. Try a lower minimum or clear the tag.",
    )


@app.post("/upload_image")
def upload_image():
    upload = request.files.get("picture")
    if not upload or not upload.filename:
        return "Choose a JPG or PNG picture.", 400
    raw = upload.read()
    upload.seek(0)
    try:
        with Image.open(upload, formats=["PNG", "JPEG"]) as image:
            if image.width * image.height > 24_000_000:
                return "Choose a picture under 24 megapixels.", 400
            suffix = ".png" if image.format == "PNG" else ".jpg"
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return "Choose a readable JPG or PNG picture.", 400

    vector = model.encode([rgb], normalize_embeddings=True)[0].tolist()
    name = sha256(raw).hexdigest() + suffix
    target = IMAGES / name
    relative = "images/" + name
    tags = sorted({tag.strip().casefold() for tag in request.form.get("tags", "").split(",")
                   if tag.strip()})
    automatic = suggest_tags(vector) if request.form.get("auto_tags") == "1" else []
    point_id = str(uuid5(NAMESPACE_URL, relative))
    IMAGES.mkdir(exist_ok=True)
    with closing(QdrantClient(path=str(STORE))) as client:
        previous = client.retrieve(
            collection_name="images", ids=[point_id], with_payload=True,
        )
        old = (previous[0].payload or {}) if previous else {}
        # Preserve manual tags from earlier uploads, including the old schema.
        prior_manual = old.get("manual_tags", old.get("tags", []))
        tags = sorted(set(tags) | set(prior_manual))
        payload = dict(old)
        payload.update(path=relative, tags=tags, manual_tags=tags, auto_tags=automatic)
        point = models.PointStruct(id=point_id, vector=vector, payload=payload)
        created = not target.exists()
        try:
            if created:
                target.write_bytes(raw)
            client.upsert(collection_name="images", points=[point], wait=True)
        except Exception:
            if created:
                target.unlink(missing_ok=True)
            raise
        # Auto suggestions are stored separately to avoid treating guesses as exact tags.
    message = "Picture indexed."
    if automatic:
        message += " AI-suggested tags (check these): " + ", ".join(automatic) + "."
    if tags:
        message += " Manual tags: " + ", ".join(tags) + "."
    return redirect(url_for("home", message=message))


EVALUATION_PAGE = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lens — Evaluate Search</title>{{ ui_style|safe }}</head><body>\n<header><nav class="nav"><a class="brand" href="/"><span>L</span>Lens</a><div class="navlinks"><a href="/">Search</a><a class="active" href="/evaluate">Evaluate</a></div></nav></header>\n<main><section class="hero"><div class="eyebrow">Measure your search</div><h1>How well did we find it?</h1><p class="muted">Evaluate semantic search without tag boosting or filters.</p></section>\n<section class="panel"><form method="post" action="/evaluate#report">\n<label class="field" for="evalquery">Search query</label><input id="evalquery" name="query" required maxlength="500" value="{{ query }}" placeholder="e.g. butterfly on a flower">\n<input type="hidden" name="catalog_token" value="{{ token }}">\n<h2>Choose the correct matches</h2><p class="muted">{{ catalog|length }} indexed pictures. Tick every picture that matches your query, including matching copies. Each filename counts separately.</p>\n<div class="evalgrid">{% for name in catalog %}<label class="evalcard"><img src="{{ url_for(\'static\', filename=name) }}" alt="{{ name }}" loading="lazy"><span class="check"><input type="checkbox" name="relevant" value="{{ name }}" {% if name in selected %}checked{% endif %}><span class="filename" title="{{ name }}">{{ name }}</span></span></label>{% endfor %}</div>\n<p class="hint">Mark the correct matches before running. A few pictures and one query are a demonstration, not a reliable benchmark.</p><button class="primary">Run evaluation</button></form></section>\n{% if report %}<section id="report"><div class="resultshead"><h2>Results for: {{ report.query }}</h2><span class="muted">{{ report.returned|length }} results returned</span></div>\n<div class="metricgrid"><div class="metric"><strong>Precision@5: {{ "%.1f"|format(report.precision_at_5 * 100) }}%</strong><span>{{ report.hits }} correct / 5 result slots</span></div><div class="metric"><strong>Recall@5: {{ "%.1f"|format(report.recall_at_5 * 100) }}%</strong><span>{{ report.hits }} found / {{ report.relevant|length }} marked relevant</span></div><div class="metric"><strong>{{ "%.3f"|format(report.search_seconds) }} s</strong><span>Search time</span></div></div>\n<p class="hint">Precision@5 always divides by 5, even with fewer results. Search time includes query encoding and database retrieval; it excludes model startup, annotation and page rendering.</p>\n<div class="panel tablewrap"><table><thead><tr><th>Rank</th><th>Picture</th><th>Relevance</th><th>Similarity</th></tr></thead><tbody>{% for row in report.returned %}<tr><td>{{ loop.index }}</td><td class="filenamecell">{{ row.name }}</td><td class="{{ \'good\' if row.name in selected else \'bad\' }}">{{ \'Correct match\' if row.name in selected else \'Not relevant\' }}</td><td>{{ "%.3f"|format(row.score) }}</td></tr>{% endfor %}</tbody></table></div><p class="notice">{{ save_message }}</p></section>{% endif %}\n<footer>Lens · Evaluation runs are saved locally in your project.</footer></main>{{ ui_script|safe }}</body></html>\n'


def retrieval_metrics(returned_names, relevant_names, k=5):
    if k < 1:
        raise ValueError("k must be positive.")
    relevant = set(relevant_names)
    if not relevant:
        raise ValueError("Mark at least one relevant picture.")
    returned = list(dict.fromkeys(returned_names))[:k]
    hits = len(set(returned) & relevant)
    return dict(hits=hits, precision_at_5=hits / k,
                recall_at_5=hits / len(relevant))


def evaluation_catalog():
    with closing(QdrantClient(path=str(STORE))) as client:
        if not client.collection_exists("images"):
            raise ValueError("Build the image index before evaluating.")
        names, offset = set(), None
        while True:
            points, offset = client.scroll(
                collection_name="images", limit=100, offset=offset,
                with_payload=True, with_vectors=False,
            )
            for point in points:
                path = (point.payload or {}).get("path")
                if path:
                    names.add(Path(path).name)
            if len(names) > 200:
                raise ValueError("This manual evaluation page supports up to 200 indexed pictures.")
            if offset is None:
                break
    if not names:
        raise ValueError("Upload and index pictures before evaluating.")
    if any(not (IMAGES / name).is_file() for name in names):
        raise ValueError("Some indexed picture files are missing. Restore them before marking relevance.")
    catalog = sorted(names)
    token = sha256(json.dumps(catalog).encode()).hexdigest()
    return catalog, token


@app.route("/evaluate", methods=["GET", "POST"])
def evaluate():
    try:
        catalog, token = evaluation_catalog()
    except ValueError as error:
        return str(error), 400
    query, selected, report, save_message = "", set(), None, ""
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        selected = set(request.form.getlist("relevant"))
        if not query or len(query) > 500:
            return "Enter a query between 1 and 500 characters.", 400
        if request.form.get("catalog_token") != token:
            return "The indexed collection changed. Reopen /evaluate and mark the pictures again.", 400
        if not selected or not selected.issubset(catalog):
            return "Tick at least one correct picture from the gallery.", 400
        started = perf_counter()
        vector = model.encode(query, prompt_name="retrieval.query",
                              normalize_embeddings=True).tolist()
        photos = retrieve_photos(vector, dict(limit=5, minimum=-1.0,
                                              tag="", source="manual"))
        seconds = perf_counter() - started
        metrics = retrieval_metrics([photo["name"] for photo in photos], selected)
        report = dict(
            timestamp=datetime.now(timezone.utc).isoformat(), query=query,
            mode="text_to_image_semantic_no_tags", model="jinaai/jina-clip-v2",
            dimensions=1024, k=5, corpus=catalog, corpus_token=token,
            relevant=sorted(selected), returned=photos,
            search_seconds=seconds, **metrics,
        )
        try:
            with (ROOT / "evaluations.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(report, ensure_ascii=False) + "\n")
            save_message = "Saved this run to evaluations.jsonl in your project folder."
        except OSError:
            save_message = "Results calculated, but the report file could not be saved."
    return render_template_string(EVALUATION_PAGE, catalog=catalog, token=token,
                                  query=query, selected=selected, report=report,
                                  save_message=save_message)


if __name__ == "__main__":
    app.run(port=5000, debug=False, threaded=False)
