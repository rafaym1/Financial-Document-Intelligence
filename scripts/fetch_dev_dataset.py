import time
from pathlib import Path

import requests

USER_AGENT = "Financial Document Intelligence (learning project) rafay.mustafa@example.com"
TICKER = "PLNT"
FILINGS_PER_FORM = 2
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dev_vdr"

CATEGORY_BY_FORM = {
    "10-K": "3 - Accounts",
    "DEF 14A": "1 - Corporate Matters",
}


def get_cik(ticker: str) -> str:
    """Look up a company's 10-digit CIK from its ticker via SEC's company_tickers.json."""
    response = requests.get("https://www.sec.gov/files/company_tickers.json", headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    for entry in response.json().values():
        if entry["ticker"] == ticker:
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker} not found")


def fetch_filings(cik: str) -> list[dict]:
    """Fetch up to FILINGS_PER_FORM of each form type in CATEGORY_BY_FORM from SEC EDGAR's submissions API."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    recent = response.json()["filings"]["recent"]

    filings = []
    counts: dict[str, int] = {}
    for i in range(len(recent["form"])):
        form = recent["form"][i]
        if form not in CATEGORY_BY_FORM:
            continue
        if counts.get(form, 0) >= FILINGS_PER_FORM:
            continue
        counts[form] = counts.get(form, 0) + 1
        filings.append(
            {
                "form": form,
                "accession_number": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
                "filing_date": recent["filingDate"][i],
            }
        )
    return filings


def download_filing(cik: str, filing: dict) -> Path:
    """Download one filing's primary document into its mapped VDR category folder."""
    accession_no_dashes = filing["accession_number"].replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/{filing['primary_document']}"

    dest_dir = DATA_DIR / CATEGORY_BY_FORM[filing["form"]]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{filing['filing_date']}_{filing['form'].replace(' ', '_')}_{filing['primary_document']}"

    response = requests.get(url, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    dest_path.write_bytes(response.content)
    time.sleep(0.2)
    return dest_path


def main():
    cik = get_cik(TICKER)
    filings = fetch_filings(cik)
    for filing in filings:
        dest_path = download_filing(cik, filing)
        print(f"Saved {dest_path}")


if __name__ == "__main__":
    main()
