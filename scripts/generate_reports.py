from pathlib import Path

from fdi.knowledge_base import load_category_facts
from fdi.reports import build_metric_series, plot_metric_trend

OUTPUT_DIR = Path("data/output/reports")

TARGET_METRICS = ["Total Revenue", "Adjusted EBITDA", "Net income", "Total assets"]


def main():
    records = load_category_facts("3 - Accounts")
    facts = [fact for record in records for fact in record["facts"]]

    for metric_name in TARGET_METRICS:
        series = build_metric_series(facts, metric_name)
        if len(series) < 2:
            print(f"Skipping '{metric_name}' -- only {len(series)} data point(s), not enough for a trend")
            continue
        output_path = OUTPUT_DIR / f"{metric_name.lower().replace(' ', '_')}.png"
        plot_metric_trend(metric_name, series, output_path)
        print(f"Saved {output_path} ({series})")


if __name__ == "__main__":
    main()
