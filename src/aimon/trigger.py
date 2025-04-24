import os
import time
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
            if name and os.path.splitext(name)[-1] in ext:
                list_files.append(link)
    finally:
        driver.quit()  # Close the browser
    
    return list_files


def download(download_page_link, last_file, new_file):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(download_page_link)
    # Wait until the wrapper div is populated
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "wrapper"))
    )
    file = driver.find_elements(By.TAG_NAME, "a")[-1]
    pc_link = file.get_attribute("href")  # Get the full URL
    print(f"Downloading the point cloud: {pc_link}")


    # Check if the file already exists
    if os.path.exists(last_file):
        # If it exists, change its name to last_point_cloud.las
        os.rename(new_file, last_file)

    with urlopen(pc_link) as in_stream, open(new_file, 'wb') as out_file:
        copyfileobj(in_stream, out_file)

    #urlretrieve(pc_link)

def main(json_path, last_file, new_file):
    # Get the initial file list
    previous_files = get_file_list()

    while True:
        time.sleep(CHECK_INTERVAL)
        current_files = get_file_list()
        
        # Detect new files
        added_file = set(current_files) - set(previous_files)
        deleted_file = set(previous_files) - set(current_files)
        
        added_file = current_files[0]
        if added_file:
            print(f"Added file detected: {added_file}")
            # Trigger some function
            download(added_file, last_file, new_file)
            os.system(f"python main.py -c \"{json_path}\" -f \"{last_file}\" \"{new_file}\"")
        if deleted_file:
            print(f"Deleted file detected: {deleted_file}")

        previous_files = current_files
        #break

if __name__ == "__main__":
    ext = [".las", ".laz", ".ply", ".txt", ".xyz"] # File extensions the user is searching for
    download_dir = r"/home/william/Downloads/" # Directory to download the files to
    FOLDER_URL = r'https://heibox.uni-heidelberg.de/d/8c6f08c92a1f48978fb4/files/' # URL to the server containing the files (heiBOX/Bibliothèques/3DGeo_Exchange/Will)
    CHECK_INTERVAL = 3  # seconds

    json_path = r"/home/william/Documents/DATA/Obergurgl/aimon_configs/Obergurgl_dev.json" # Path to the json file
    last_file = os.path.join(download_dir, "last_point_cloud.las")
    new_file = os.path.join(download_dir, "new_point_cloud.las")

    main(json_path, last_file, new_file)