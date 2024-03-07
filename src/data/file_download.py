import urllib.request
import os


def download_file(url, filename):
    if os.path.exists(filename):
        print(f"The file '{filename}' already exists. Skipping download.")
    else:
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded '{filename}' successfully.")
        except Exception as e:
            print(f"Failed to download '{filename}'. Error: {e}")


if __name__ == "__main__":
    # Random paper from arxiv
    file_url = "https://arxiv.org/pdf/2403.02484.pdf"
    file_name = "test.pdf"
    download_file(file_url, file_name)
