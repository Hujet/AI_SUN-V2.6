import requests
import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple

class HelioviewerClient:
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("HELIOVIEWER_TOKEN", "")
        self.base_url = "https://api.helioviewer.org/v2/"
        self.image_dir = os.path.join(os.path.dirname(__file__), "..", "resource", "data")

    def getJP2Image(self, date: str, sourceId: int, jpip: bool = False) -> bytes:
        endpoint = "getJP2Image"
        params = {
            "date": date,
            "sourceId": sourceId,
            "jpip": str(jpip).lower()
        }
        if self.api_token:
            params["token"] = self.api_token
        response = requests.get(self.base_url + endpoint, params=params)
        response.raise_for_status()
        return response.content

    def getThumbnail(self, date: str, sourceId: int, scale: float = 0.5) -> bytes:
        endpoint = "getThumbnail"
        params = {
            "date": date,
            "sourceId": sourceId,
            "scale": scale
        }
        if self.api_token:
            params["token"] = self.api_token
        response = requests.get(self.base_url + endpoint, params=params)
        response.raise_for_status()
        return response.content

    def saveJP2(self, data: bytes, filename: str) -> str:
        filepath = os.path.join(self.image_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(data)
        return filepath

    def saveThumbnail(self, data: bytes, filename: str) -> str:
        filepath = os.path.join(self.image_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(data)
        return filepath

    def getImageList(self, startDate: str, endDate: str, sourceId: int) -> List[Dict]:
        endpoint = "getImageList"
        params = {
            "startDate": startDate,
            "endDate": endDate,
            "sourceId": sourceId
        }
        if self.api_token:
            params["token"] = self.api_token
        response = requests.get(self.base_url + endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def getDataSources(self) -> List[Dict]:
        endpoint = "getDataSources"
        if self.api_token:
            params = {"token": self.api_token}
            response = requests.get(self.base_url + endpoint, params=params)
        else:
            response = requests.get(self.base_url + endpoint)
        response.raise_for_status()
        return response.json()

    def findActiveRegions(self, startDate: str, endDate: str) -> List[Dict]:
        endpoint = "findActiveRegions"
        params = {
            "startDate": startDate,
            "endDate": endDate
        }
        if self.api_token:
            params["token"] = self.api_token
        response = requests.get(self.base_url + endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def getMostRecentImage(self, sourceId: int) -> Dict:
        endpoint = "getMostRecentImage"
        params = {"sourceId": sourceId}
        if self.api_token:
            params["token"] = self.api_token
        response = requests.get(self.base_url + endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def downloadMagnetogram(self, date: str, output_filename: Optional[str] = None) -> Tuple[str, bytes]:
        sourceId = 10
        data = self.getJP2Image(date, sourceId)
        if output_filename is None:
            output_filename = f"magnetogram_{date.replace(':', '_').replace('-', '')}.jp2"
        filepath = self.saveJP2(data, output_filename)
        return filepath, data

    def downloadEUVImage(self, date: str, wavelength: int = 171, output_filename: Optional[str] = None) -> Tuple[str, bytes]:
        source_map = {
            94: 15,
            131: 16,
            171: 17,
            193: 18,
            211: 19,
            304: 20,
            335: 21,
            1600: 22
        }
        sourceId = source_map.get(wavelength, 17)
        data = self.getJP2Image(date, sourceId)
        if output_filename is None:
            output_filename = f"euv_{wavelength}_{date.replace(':', '_').replace('-', '')}.jp2"
        filepath = self.saveJP2(data, output_filename)
        return filepath, data

    def downloadThumbnail(self, date: str, image_type: str = "magnetogram", scale: float = 0.5) -> Tuple[str, bytes]:
        if image_type == "magnetogram":
            sourceId = 10
        elif image_type == "euv171":
            sourceId = 17
        elif image_type == "euv193":
            sourceId = 18
        else:
            sourceId = 10
        data = self.getThumbnail(date, sourceId, scale)
        filename = f"thumb_{image_type}_{date.replace(':', '_').replace('-', '')}.jpg"
        filepath = self.saveThumbnail(data, filename)
        return filepath, data

def get_source_id_map() -> Dict[str, int]:
    return {
        "HMI_Magnetogram": 10,
        "HMI_Continuum": 8,
        "AIA_94": 15,
        "AIA_131": 16,
        "AIA_171": 17,
        "AIA_193": 18,
        "AIA_211": 19,
        "AIA_304": 20,
        "AIA_335": 21,
        "AIA_1600": 22
    }

def format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def parse_date(date_str: str) -> datetime:
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y%m%d_%H%M%S"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date string: {date_str}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python helioviewer_client.py <date> [type]")
        print("  date: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
        print("  type: magnetogram (default), euv171, euv193, thumbnail")
        sys.exit(1)

    date = sys.argv[1]
    img_type = sys.argv[2] if len(sys.argv) > 2 else "magnetogram"

    client = HelioviewerClient()
    try:
        if img_type == "thumbnail":
            filepath, data = client.downloadThumbnail(date, image_type="magnetogram")
            print(f"Thumbnail saved to: {filepath}")
        elif img_type.startswith("euv"):
            wavelength = int(img_type.replace("euv", ""))
            filepath, data = client.downloadEUVImage(date, wavelength=wavelength)
            print(f"EUV image saved to: {filepath}")
        else:
            filepath, data = client.downloadMagnetogram(date)
            print(f"Magnetogram saved to: {filepath}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)