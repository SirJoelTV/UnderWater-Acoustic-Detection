import os
import requests
from bs4 import BeautifulSoup

# URL of the page with whale sound downloads
url = "https://whoicf2.whoi.edu/science/B/whalesounds/bestOf.cfm?code=BB1A"

# Folder to save the downloaded files
save_folder = "D:\Main Project\Data\Beluga, White Whale"
os.makedirs(save_folder, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

# Fetch the webpage content
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all download links (anchor tags with text 'Download')
download_links = soup.find_all('a', string='Download')

print(f"Found {len(download_links)} download links.")

for idx, link in enumerate(download_links, 1):
    href = link.get('href')
    if not href.startswith('http'):
        # Build full URL
        file_url = requests.compat.urljoin(url, href)
    else:
        file_url = href

    filename = file_url.split('/')[-1]
    if not filename.lower().endswith('.wav'):
        # Add .wav extension if missing (sometimes URLs end with query params)
        filename += '.wav'

    print(f"Downloading {idx}/{len(download_links)}: {filename} ...")

    try:
        file_response = requests.get(file_url, headers=headers)
        # Simple verification of WAV file signature: 'RIFF' at start
        if file_response.content[:4] == b'RIFF':
            file_path = os.path.join(save_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(file_response.content)
        else:
            print(f"Warning: {filename} does not appear to be a valid wav file. Skipping.")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

print(f"Download complete. Files saved in: {save_folder}")
