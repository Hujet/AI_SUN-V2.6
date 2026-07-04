import os
import sys
import json
import csv
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helioviewer_client import HelioviewerClient, format_date, parse_date

class SolarDataAcquirer:
    def __init__(self, api_token: Optional[str] = None):
        self.client = HelioviewerClient(api_token)
        self.data_dir = Path(__file__).parent.parent / "resource" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def collectHistoricalEvents(self, start_date: str, end_date: str, target_count: int = 20) -> List[Dict]:
        print(f"Searching for active regions from {start_date} to {end_date}...")
        regions = self.client.findActiveRegions(start_date, end_date)
        print(f"Found {len(regions)} active regions")

        selected = []
        for region in regions[:target_count]:
            region_info = {
                "id": region.get("id", ""),
                "name": region.get("name", ""),
                "latitude": region.get("latitude", ""),
                "longitude": region.get("longitude", ""),
                "date_start": region.get("date_start", ""),
                "date_end": region.get("date_end", ""),
                "classification": region.get("classification", ""),
                "area": region.get("area", 0),
                "numspots": region.get("numspots", 0)
            }
            selected.append(region_info)
        return selected

    def downloadActiveRegionData(self, region: Dict, output_dir: Optional[Path] = None) -> Dict:
        if output_dir is None:
            output_dir = self.data_dir / region.get("name", f"region_{region['id']}")
        output_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "region_id": region.get("id"),
            "region_name": region.get("name"),
            "downloads": []
        }

        if region.get("date_start"):
            date = region["date_start"]
        else:
            date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        try:
            magnetogram_path, _ = self.client.downloadMagnetogram(
                date,
                f"AR{region.get('name', region['id'])}_HMI.jp2"
            )
            result["downloads"].append({"type": "magnetogram", "path": magnetogram_path})
            print(f"  Downloaded magnetogram: {magnetogram_path}")
        except Exception as e:
            print(f"  Error downloading magnetogram: {e}")

        for wavelength in [171, 193]:
            try:
                euv_path, _ = self.client.downloadEUVImage(
                    date,
                    wavelength=wavelength,
                    output_filename=f"AR{region.get('name', region['id'])}_AIA{wavelength}.jp2"
                )
                result["downloads"].append({"type": f"euv{wavelength}", "path": euv_path})
                print(f"  Downloaded EUV {wavelength}: {euv_path}")
            except Exception as e:
                print(f"  Error downloading EUV {wavelength}: {e}")

        try:
            thumb_path, _ = self.client.downloadThumbnail(date, image_type="magnetogram")
            result["downloads"].append({"type": "thumbnail", "path": thumb_path})
            print(f"  Downloaded thumbnail: {thumb_path}")
        except Exception as e:
            print(f"  Error downloading thumbnail: {e}")

        return result

    def batchDownload(self, regions: List[Dict], output_base: Optional[Path] = None) -> List[Dict]:
        results = []
        for i, region in enumerate(regions, 1):
            print(f"\n[{i}/{len(regions)}] Processing region: {region.get('name', region['id'])}")
            try:
                result = self.downloadActiveRegionData(region, output_base)
                results.append(result)
            except Exception as e:
                print(f"Error processing region: {e}")
                results.append({
                    "region_id": region.get("id"),
                    "region_name": region.get("name"),
                    "error": str(e),
                    "downloads": []
                })
        return results

    def saveResults(self, results: List[Dict], filename: str = "download_results.json"):
        filepath = self.data_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {filepath}")
        return filepath

    def saveRegionsList(self, regions: List[Dict], filename: str = "regions_list.csv"):
        filepath = self.data_dir / filename
        if not regions:
            print("No regions to save")
            return None
        keys = list(regions[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(regions)
        print(f"Regions list saved to: {filepath}")
        return filepath

    def downloadKnownEvent(self, event_date: str, event_name: str, event_id: Optional[str] = None) -> Dict:
        event_dir = self.data_dir / f"event_{event_name}_{event_date[:10].replace('-', '')}"
        event_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "event_name": event_name,
            "event_date": event_date,
            "downloads": []
        }

        source_files = {
            "magnetogram": ("magnetogram", 10),
            "euv171": ("euv171", 17),
            "euv193": ("euv193", 18),
            "euv211": ("euv211", 19),
        }

        for name, (img_type, source_id) in source_files.items():
            try:
                filename = f"{event_name}_{event_date[:10].replace('-', '')}_{name}.jp2"
                filepath, _ = self.client.downloadThumbnail(event_date, image_type=img_type)
                new_path = event_dir / filename
                os.rename(filepath, new_path)
                result["downloads"].append({"type": name, "path": str(new_path)})
                print(f"  Downloaded {name}: {new_path}")
            except Exception as e:
                print(f"  Error downloading {name}: {e}")

        metadata_file = event_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({
                "event_name": event_name,
                "event_date": event_date,
                "event_id": event_id,
                "download_time": datetime.now().isoformat()
            }, f, indent=2)
        result["metadata"] = str(metadata_file)

        return result

def createSampleDataset(target_dir: Path, num_samples: int = 10):
    target_dir.mkdir(parents=True, exist_ok=True)
    sample_events = [
        {"date": "2024-05-10T12:00:00", "name": "X8.7_flare_AR13664", "noaa_id": "13664"},
        {"date": "2024-05-08T14:00:00", "name": "X2.2_flare_AR13663", "noaa_id": "13663"},
        {"date": "2024-05-05T18:00:00", "name": "M4.5_flare_AR13661", "noaa_id": "13661"},
        {"date": "2023-12-15T06:00:00", "name": "X2.8_flare_AR13510", "noaa_id": "13510"},
        {"date": "2023-11-28T14:00:00", "name": "M9.8_flare_AR13498", "noaa_id": "13498"},
    ]
    metadata = []
    for i, event in enumerate(sample_events[:min(num_samples, len(sample_events))]):
        event_record = {
            "index": i + 1,
            "event_name": event["name"],
            "event_date": event["date"],
            "noaa_id": event["noaa_id"],
            "hale_class_expected": "Beta-Gamma",
            "file_magnetogram": f"magnetogram_{event['noaa_id']}.jpg",
            "file_euv171": f"euv171_{event['noaa_id']}.jpg",
            "file_euv193": f"euv193_{event['noaa_id']}.jpg",
            "notes": f"Please download from Helioviewer.org at {event['date']}"
        }
        metadata.append(event_record)

    metadata_file = target_dir / "sample_dataset_metadata.csv"
    keys = list(metadata[0].keys())
    with open(metadata_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metadata)
    print(f"Sample dataset template saved to: {metadata_file}")
    return metadata_file

def main():
    parser = argparse.ArgumentParser(description="Solar Data Acquisition Tool")
    parser.add_argument("--token", "-t", type=str, help="Helioviewer API token")
    parser.add_argument("--start", "-s", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", "-e", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--count", "-c", type=int, default=20, help="Number of samples to collect")
    parser.add_argument("--batch", "-b", action="store_true", help="Batch download mode")
    parser.add_argument("--create-template", action="store_true", help="Create sample dataset template")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory")

    args = parser.parse_args()

    if args.create_template:
        output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent / "resource" / "data"
        createSampleDataset(output_dir)
        return

    acquirer = SolarDataAcquirer(api_token=args.token)

    if args.start and args.end:
        regions = acquirer.collectHistoricalEvents(args.start, args.end, args.count)
        if args.batch:
            results = acquirer.batchDownload(regions)
            acquirer.saveResults(results)
        acquirer.saveRegionsList(regions)
        print(f"\nCollected {len(regions)} active regions")
    else:
        print("No date range specified. Use --start and --end options.")
        print("Creating sample dataset template instead...")
        output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent / "resource" / "data"
        createSampleDataset(output_dir)

if __name__ == "__main__":
    main()