import os
import time
import urllib
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from urllib.request import urlretrieve, urlopen

from shutil import copyfileobj

# Set up Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in background (optional)
options.add_argument("--disable-gpu")  # Helps with some issues

def get_file_list():
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(FOLDER_URL)
        
        # Wait until the wrapper div is populated
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "wrapper"))
        )

        # Extract filenames (modify based on structure)
        filenames = driver.find_elements(By.TAG_NAME, "a")
        list_files = []
        for file in filenames:
            name = file.text.strip()
            link = file.get_attribute("href")  # Get the full URL
            if name and os.path.splitext(name)[-1] in EXT:
                list_files.append(link)
    finally:
        driver.quit()  # Close the browser
    
    return list_files


def download(download_page_link):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(download_page_link)
    # Wait until the wrapper div is populated
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "wrapper"))
    )
    file = driver.find_elements(By.TAG_NAME, "a")[-1]
    pc_link = file.get_attribute("href")  # Get the full URL
    pc_link_name = urllib.parse.unquote(pc_link)
    pc_file_name = os.path.basename(pc_link_name).split('&')[0]
    print(f"Downloading the point cloud: {pc_file_name}")

    download_path = os.path.join(download_dir, pc_file_name)
    with urlopen(pc_link) as in_stream, open(download_path, 'wb') as out_file:
        copyfileobj(in_stream, out_file)
    print("Done")
    return download_path

def main(json_path):
    # Get the initial file list
    try:
        previous_files = get_file_list()
        previous_files.sort()
        last_file = urllib.parse.unquote(previous_files[-1])
        last_file = os.path.basename(last_file).split('&')[0]
        last_file = os.path.join(download_dir, last_file)
    except:
        last_file = []

    while True:
        time.sleep(CHECK_INTERVAL)
        current_files = get_file_list()
        current_files.sort()
        # Detect new files
        added_file = list(set(current_files) - set(previous_files))

        if added_file and last_file != []:
            print(f"New file detected: {added_file}")
            new_file = download(added_file[0])
            os.system(f"python src/aimon/main.py -c '{json_path}' -f '{last_file}' '{new_file}'")
            to_delete = last_file
            last_file = new_file

        elif added_file and last_file == []:
            print(f"New file detected: {added_file}")
            new_file = download(added_file[0])
            last_file = new_file

        # if "to_delete" in locals() or "to_delete" in globals():
        #     os.remove(to_delete)
        #     if to_delete in globals():
        #         del globals()[to_delete]
        #     del to_delete

        previous_files = current_files
        #break

if __name__ == "__main__":
    EXT = [".las", ".laz", ".ply", ".txt", ".xyz"] # File extensions the user is searching for
    FOLDER_URL = r'https://heibox.uni-heidelberg.de/d/8c6f08c92a1f48978fb4/files/' # URL to the server containing the files (3DGeo/3DGeo_Exchange/Will)
    CHECK_INTERVAL = 3  # seconds
    download_dir = r"/home/william/Downloads/" # Directory to download the files to

    json_path = r"/home/william/Documents/DATA/Obergurgl/aimon_configs/Obergurgl_dev.json" # Path to the json file

    main(json_path)