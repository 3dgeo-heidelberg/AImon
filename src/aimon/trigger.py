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

# File extensions the user is searching for
ext = [".las", ".laz", ".ply", ".txt", ".xyz"]

DOWNLOAD_DIR = r"/home/william/Downloads/"
FOLDER_URL = r'https://heibox.uni-heidelberg.de/d/8c6f08c92a1f48978fb4/files/'
CHECK_INTERVAL = 3

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


def download(download_page_link):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(download_page_link)
    # Wait until the wrapper div is populated
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "wrapper"))
    )
    file = driver.find_elements(By.TAG_NAME, "a")[-1]
    pc_link = file.get_attribute("href")  # Get the full URL
    print(f"Downloading the point cloud: {pc_link}")

    with urlopen(pc_link) as in_stream, open('/home/william/Downloads/new_point_cloud.las', 'wb') as out_file:
        copyfileobj(in_stream, out_file)

    #urlretrieve(pc_link)

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
        download(added_file)
    if deleted_file:
        print(f"Deleted file detected: {deleted_file}")

    previous_files = current_files
    #break

###############################################

# class ExampleHandler(FileSystemEventHandler):
#     def on_created(self, event): # when file is created
#         t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
#         print(f"File created: {event.src_path}\n{t}")

#     def on_deleted(self, event): # when file is deleted
#         t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
#         print(f"File deleted: {event.src_path}\n{t}")

# observer = Observer()
# observer.schedule(ExampleHandler(), path=FOLDER_URL)
# observer.start()

# # sleep until keyboard interrupt, then stop + rejoin the observer
# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     observer.stop()
# observer.join()