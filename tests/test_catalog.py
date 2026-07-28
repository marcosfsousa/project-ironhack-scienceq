# tests/test_catalog.py

"""
Coverage for the repo's only GCS call site, added alongside the
google-cloud-storage 2.18.2 → 3.13.0 major bump (#64).

Before this file the path at api/catalog.py:48-52 was untested, and untested in
a way that was easy to miss: `from google.cloud import storage` sits *inside*
`_corpus_videos` rather than at module scope, so importing api.catalog never
touched the package. A green suite on a google-cloud-storage PR was therefore
measuring the resolve and nothing else — which is exactly the reassurance a
major version bump must not be given for free.

Two different things are asserted here, and the split is the point:

  - `test_gcs_api_surface` reaches the *real* installed package. It is the
    regression guard proper: a future major that drops `Blob.download_as_text`
    or renames `Client.bucket` fails here, at the version bump, instead of at
    the next GET /api/catalog in production.
  - the rest inject a fake client and never import a credential. They cover the
    parsing and the degrade-to-live fallback, which are this repo's code and
    would keep passing against a hollowed-out client — hence the first test.

Neither touches the network. CI runs without secrets on purpose
(.github/workflows/ci.yml), and a test that acquired a live GCS dependency
would fail there while passing on a developer machine with ADC configured. The
live read stays a post-deploy check; it is not something this file can claim.

There is deliberately no `pytest.importorskip` on google.cloud.storage. The
package is in requirements.txt, which CI installs and requirements-dev.txt
inherits via `-r`, so any environment that can run this suite has it — and a
skip would turn the one assertion that catches a removed method into silence
in precisely the environment where it went missing.
"""

import json

import pytest

from api import catalog


# metadata.json in the bucket is the `{"videos": {...}}` form, so that is what
# the fake returns — the bare-dict and list forms `_parse_metadata` also accepts
# are covered by their own case below rather than folded in here.
METADATA = {
    "videos": {
        "jAhjPd4uNFY": {
            "title":    "Genetic Engineering Will Change Everything Forever – CRISPR",
            "channel":  "Kurzgesagt",
            "topic":    "biology",
            "duration": "16:03",
        },
        "QOK6PmGxSAI": {
            "title":    "The Immune System Explained",
            "channel":  "Kurzgesagt",
            "topic":    "biology",
            "duration": "07:12",
        },
    }
}


class _FakeBlob:
    def __init__(self, payload):
        self._payload = payload

    def download_as_text(self):
        return json.dumps(self._payload)


class _FakeBucket:
    def __init__(self, payload, calls):
        self._payload = payload
        self._calls   = calls

    def blob(self, name):
        self._calls["blob"] = name
        return _FakeBlob(self._payload)


class _FakeClient:
    """Stands in for storage.Client(), which would otherwise want credentials."""

    payload = METADATA
    calls: dict = {}

    def __init__(self):
        type(self).calls.clear()

    def bucket(self, name):
        type(self).calls["bucket"] = name
        return _FakeBucket(type(self).payload, type(self).calls)


@pytest.fixture
def fake_gcs(monkeypatch):
    """Swap storage.Client on the real module `_corpus_videos` imports from.

    The lazy import inside the function resolves `storage` to the same module
    object this patches, so the substitution lands whether or not anything else
    in the session has imported it already.
    """
    from google.cloud import storage

    monkeypatch.setattr(storage, "Client", _FakeClient)
    _FakeClient.payload = METADATA
    return _FakeClient


def test_gcs_api_surface():
    """The four names api/catalog.py:50-52 resolve against still exist.

    Asserted against the installed package rather than a stub, so this fails on
    a major that removes or renames one of them. Attributes are checked on the
    classes, not on an instance, because instantiating Client() needs
    credentials this suite deliberately does not have.
    """
    from google.cloud import storage

    assert callable(storage.Client)
    assert callable(storage.Client.bucket)
    assert callable(storage.Bucket.blob)
    assert callable(storage.Blob.download_as_text)


def test_corpus_videos_reads_the_expected_object(fake_gcs):
    """The bucket and blob names are the ones the docstring promises."""
    catalog._corpus_videos()

    assert fake_gcs.calls["bucket"] == "scienceq-data"
    assert fake_gcs.calls["blob"] == "metadata.json"


def test_corpus_videos_shape(fake_gcs):
    videos = catalog._corpus_videos()

    assert len(videos) == 2
    crispr = next(v for v in videos if v["video_id"] == "jAhjPd4uNFY")
    assert crispr == {
        "video_id": "jAhjPd4uNFY",
        "title":    "Genetic Engineering Will Change Everything Forever – CRISPR",
        "channel":  "Kurzgesagt",
        "topic":    "biology",
        "duration": "16:03",
        "url":      "https://www.youtube.com/watch?v=jAhjPd4uNFY",
        "source":   "corpus",
    }


def test_corpus_videos_fills_missing_fields(fake_gcs):
    """A sparse entry yields empty strings, not KeyError — the payload is a
    contract the frontend reads, so a partial metadata.json must not 500."""
    fake_gcs.payload = {"videos": {"abc123": {"title": "Only a title"}}}

    (video,) = catalog._corpus_videos()

    assert video["channel"] == ""
    assert video["topic"] == ""
    assert video["duration"] == ""
    assert video["url"] == "https://www.youtube.com/watch?v=abc123"


@pytest.mark.parametrize(
    "raw",
    [
        {"videos": {"abc123": {"title": "T"}}},          # nested under "videos"
        {"abc123": {"title": "T"}},                      # bare id -> entry map
        [{"video_id": "abc123", "title": "T"}],          # already a list
    ],
    ids=["videos-key", "bare-dict", "list"],
)
def test_parse_metadata_accepts_all_three_forms(raw):
    (entry,) = catalog._parse_metadata(raw)

    assert entry["video_id"] == "abc123"
    assert entry["title"] == "T"


def test_get_catalog_degrades_when_gcs_fails(monkeypatch):
    """A failed corpus read leaves the live half intact.

    This is the fallback the 3.0 exception move relies on. 3.0 relocated
    InvalidResponse and DataCorruption from google.resumable_media.common to
    google.cloud.storage.exceptions; get_catalog catches bare Exception, so
    neither class needs naming here and the endpoint degrades either way. The
    raise is a plain Exception precisely to assert that breadth — narrowing the
    handler to specific storage classes would break this test, which is the
    warning intended.
    """
    live = [{"video_id": "live1", "title": "Live", "source": "live"}]
    monkeypatch.setattr(catalog, "_live_videos", lambda: live)
    monkeypatch.setattr(
        catalog, "_corpus_videos", lambda: (_ for _ in ()).throw(Exception("GCS down"))
    )

    assert catalog.get_catalog() == live


def test_get_catalog_live_wins_over_corpus(monkeypatch):
    """Both sources hold the same video_id; the live entry is the one kept."""
    monkeypatch.setattr(
        catalog, "_live_videos",
        lambda: [{"video_id": "dupe", "title": "Fresh", "source": "live"}],
    )
    monkeypatch.setattr(
        catalog, "_corpus_videos",
        lambda: [
            {"video_id": "dupe", "title": "Stale", "source": "corpus"},
            {"video_id": "only-corpus", "title": "Kept", "source": "corpus"},
        ],
    )

    merged = catalog.get_catalog()

    assert [v["video_id"] for v in merged] == ["dupe", "only-corpus"]
    assert merged[0]["title"] == "Fresh"
