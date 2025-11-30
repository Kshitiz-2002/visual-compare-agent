import requests

API = "http://127.0.0.1:8000/api/compare-images"

def run_test():
    files = {
        "image1": open("example_images/image1.png", "rb"),
        "image2": open("example_images/image2.png", "rb"),
    }
    resp = requests.post(API, files=files)
    print("status:", resp.status_code)
    print("json:", resp.json())

if __name__ == "__main__":
    run_test()
