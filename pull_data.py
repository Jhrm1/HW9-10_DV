# Download DESI tables from NOIRLab Data Lab.
from pathlib import Path
import time

import pandas as pd
from requests.exceptions import HTTPError
from astroquery.utils.tap.core import TapPlus

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def get_results_with_retry(job):
    transient_errors = ("Error 502:", "Error 503:", "Error 504:")
    for attempt in range(6):
        try:
            return job.get_results()
        except HTTPError as error:
            # Astroquery formats TAP HTTP errors as "Error <status>:" messages.
            if not str(error).startswith(transient_errors) or attempt == 5:
                raise
            delay = 2 ** attempt
            print(
                f"Temporary TAP gateway error; "
                f"retrying in {delay}s ({attempt + 1}/5)."
            )
            time.sleep(delay)


def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Supplying host and TAP context separately avoids duplicating the /tap path.
    tap = TapPlus(
        host="datalab.noirlab.edu",
        tap_context="/tap",
        default_protocol_is_https=True,
    )

    def run_query(query):
        job = tap.launch_job_async(
            query,
            output_format="csv",
            background=True,
            autorun=False,
        )
        results = get_results_with_retry(job)
        phase = job.get_phase()
        if phase != "COMPLETED":
            raise RuntimeError(f"Data Lab TAP query failed with phase {phase}.")
        return results

    qso = run_query("""
        SELECT mean_fiber_ra AS ra, mean_fiber_dec AS dec
        FROM desi_dr1.zpix
        WHERE spectype = 'QSO'
          AND mean_fiber_ra IS NOT NULL
          AND mean_fiber_dec IS NOT NULL
    """)
    totals = run_query("""
        SELECT healpix, COUNT(*) AS n_spec
        FROM desi_dr1.zpix
        WHERE healpix IS NOT NULL
        GROUP BY healpix
    """)

    qso.to_pandas().to_csv(DATA_DIR / "desi_qso.csv", index=False)
    totals.to_pandas().to_csv(DATA_DIR / "desi_pixel_totals.csv", index=False)


if __name__ == "__main__":
    download_data()