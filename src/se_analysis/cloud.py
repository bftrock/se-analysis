"""Pulling a site's data out of NRG Cloud, a day at a time or as one export job.

Two routes to the same data, because NRG Cloud offers two endpoints.

export_days() uses the synchronous export endpoint, where the POST returns the
zipped file itself -- so asking for a month blocks for as long as the server
needs to build a month. Asking for a day at a time keeps every call short, and
makes the download resumable: an interrupted range leaves the days it did fetch
on disk, and re-running picks up from there rather than starting over.

export_range() uses the export job endpoint, which builds the export server-side
while we poll for its status, then downloads the finished file in one piece. One
file for the whole range, and no practical ceiling on how much a single request
may ask for, at the cost of that resumability: an interrupted job leaves nothing
on disk, and re-running starts the export over.

So: export_days() while a range is still growing a day at a time, export_range()
for a settled one -- a closed month, or a year -- where one file beats thirty.

Credentials come from CLIENT_ID and CLIENT_SECRET, which belong in .env at the
repository root; see .env.example. nrgpy caches the bearer token in a pickle
keyed to the client id, so a CloudExport per day costs no extra tokens -- which
matters, because an account may only mint ten of them a day.
"""

import os
from pathlib import Path

import nrgpy
import pandas as pd

# nrgpy takes dates as YYYY-MM-DD, and the file it hands back is named with the
# range it covers as YYYY.MM.DD-YYYY.MM.DD
REQUEST_DATE = "%Y-%m-%d"
FILENAME_DATE = "%Y.%m.%d"


def _credentials(client_id: str | None, client_secret: str | None) -> tuple[str, str]:
    """The given credentials, or the ones load_dotenv() put in the environment.

    Checked here because nrgpy does not fail on missing credentials: CloudApi
    prints an access notice, skips the token request, and never assigns the
    `headers` it later reads, so the real symptom is an AttributeError raised
    from inside the export -- which says nothing about the credentials.
    """
    client_id = client_id or os.getenv("CLIENT_ID")
    client_secret = client_secret or os.getenv("CLIENT_SECRET")

    if not client_id or not client_secret:
        given = (("CLIENT_ID", client_id), ("CLIENT_SECRET", client_secret))
        missing = ", ".join(name for name, value in given if not value)
        raise RuntimeError(
            f"No NRG Cloud credentials: {missing} is unset. Call dotenv.load_dotenv() "
            "first, or pass client_id/client_secret. See .env.example for where to "
            "get them."
        )

    return client_id, client_secret


def _existing(out_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> Path | None:
    """An already-downloaded export covering exactly [start, end], if there is one.

    Anchored on both ends of the range, so a single-day export is not mistaken
    for a longer one that merely begins on that day, nor the other way round.
    """
    stamps = f"{start.strftime(FILENAME_DATE)}-{end.strftime(FILENAME_DATE)}"
    matches = sorted(out_dir.glob(f"*_{stamps}.*"))

    return matches[0] if matches else None


def export_days(
    site_id: int,
    start: str,
    end: str,
    out_dir: Path | str,
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    export_type: str = "measurements",
    skip_existing: bool = True,
    **kwargs,
) -> list[Path]:
    """Export one file per day over [start, end] into out_dir.

    Returns a path per day of the range that is now on disk, whether this call
    downloaded it or a previous one did, so the same arguments describe the same
    files however many times you run it. Days the API refused are reported and
    left out; the rest of the range still downloads.

    `site_id` is the NRG Cloud site ID, not the site number -- pass the number
    instead and nrgpy resolves it by fetching the whole site list again on every
    single day of the range. nrgpy.CloudSites.get_siteid() resolves it once.

    Dates are whole days in the logger's local time. Extra keyword arguments go
    to nrgpy.CloudExport, so `interval` and `nec_file` work as they do there.
    """
    client_id, client_secret = _credentials(client_id, client_secret)
    out_dir = Path(out_dir)
    paths = []

    for timestamp in pd.date_range(start, end, freq="D"):
        day = timestamp.strftime(REQUEST_DATE)

        existing = _existing(out_dir, timestamp, timestamp)
        if skip_existing and existing:
            paths.append(existing)
            continue

        exporter = nrgpy.CloudExport(
            client_id=client_id,
            client_secret=client_secret,
            site_id=site_id,
            start_date=day,
            end_date=day,  # a date with no time means the whole of that day
            out_dir=str(out_dir),
            export_type=export_type,
            **kwargs,
        )

        try:
            # export() returns False when it fails and None when it works
            if exporter.export() is False:
                print(f"{day}: export failed")
                continue
        except Exception as exc:
            # a non-200 leaves nrgpy parsing the error body positionally, which
            # raises on any shape it did not expect; a day with no data leaves it
            # opening an empty zip. Neither should cost us the rest of the range.
            print(f"{day}: {type(exc).__name__}: {exc}")
            continue

        paths.append(Path(exporter.export_filepath))

    return paths


def export_range(
    site_id: int,
    start: str,
    end: str,
    out_dir: Path | str,
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    export_type: str = "measurements",
    skip_existing: bool = True,
    **kwargs,
) -> list[Path]:
    """Export all of [start, end] as one NRG Cloud export job into out_dir.

    Takes the same arguments as export_days() and returns the same thing -- the
    paths now on disk, whether this call downloaded them or a previous one did --
    so the two are interchangeable at a call site. The difference is one request
    for the whole range instead of one per day, which is what makes a range too
    large for the synchronous endpoint retrievable at all.

    Waiting on the job is nrgpy's: it prints a status line while the server
    builds the export, and a progress bar while it downloads. A job that fails to
    start, errors out, or will not download is reported and returns no paths,
    along with its job id -- an export the server finished building may still be
    there to collect even when this call could not fetch it.

    `site_id` is the NRG Cloud site ID, not the site number -- pass the number
    instead and nrgpy resolves it by fetching the whole site list first.
    nrgpy.CloudSites.get_siteid() resolves it once.

    Dates are whole days in the logger's local time. Extra keyword arguments go
    to nrgpy.CloudExportJob, so `interval`, `nec_file` and `file_format` work as
    they do there. Note that with file_format="multipleFiles" nrgpy extracts
    every file of the zip but records only the first, so the rest arrive in
    out_dir without appearing in the returned list.
    """
    client_id, client_secret = _credentials(client_id, client_secret)
    out_dir = Path(out_dir)
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    label = f"{first.strftime(REQUEST_DATE)}..{last.strftime(REQUEST_DATE)}"

    existing = _existing(out_dir, first, last)
    if skip_existing and existing:
        return [existing]

    exporter = nrgpy.CloudExportJob(
        client_id=client_id,
        client_secret=client_secret,
        site_id=site_id,
        start_date=start,
        end_date=end,
        out_dir=str(out_dir),
        export_type=export_type,
        **kwargs,
    )

    exporter.create_export_job()

    # a refused job is logged and returns, leaving nothing to monitor -- and
    # monitoring it anyway would read the job_id that was never assigned
    if not getattr(exporter, "job_id", None):
        print(f"{label}: export job not created")
        return []

    try:
        exporter.monitor_export_job(download=True)
    except Exception as exc:
        # nrgpy parses error bodies positionally, so a non-200 mid-poll raises on
        # any shape it did not expect. The job id is the only way back to an
        # export the server may have gone on to finish.
        print(f"{label}: {type(exc).__name__}: {exc} (job {exporter.job_id})")
        return []

    # assigned only by a download that got its file: a job that ended in error,
    # or a KeyboardInterrupt out of the polling loop, leaves it unset
    if not hasattr(exporter, "export_filepath"):
        status = exporter.json_response.get("status", "unknown")
        print(f"{label}: job {exporter.job_id} not downloaded (status: {status})")
        return []

    return [Path(exporter.export_filepath)]
