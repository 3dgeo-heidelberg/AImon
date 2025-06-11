import os
import time
import urllib
from selenium import webdriver
import subprocess

from aimon.helpers.utilities import read_json_file, convert_geojson_to_datamodel, get_online_file_list, download_online_file

# Set up Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in background (optional)
options.add_argument("--disable-gpu")  # Helps with some issues


def geojson_to_datamodel(geojson_name, datamodel_json_path):
    # Convert the change events GeoJSON file to a datamodel JSON file
    loaded_data = read_json_file(geojson_name)
    datamodel_json = convert_geojson_to_datamodel(loaded_data, "/4dgeo/testing_data/rockfall_monitoring_trier/Trier_RangeImage.png", 1724, 862,)

    # Save the datamodel JSON to a file
    with open(datamodel_json_path, "w") as outfile:
        outfile.write(datamodel_json)


def main(config_json_path, datamodel_json_path, download_dir):
    with open(".temp/current_files.txt", "w") as f:
        f.write("")

    # Get the initial file list
    try:
        previous_files = get_online_file_list(FOLDER_URL, options)
        previous_files.sort()
        last_file = urllib.parse.unquote(previous_files[-1])
        last_file = os.path.basename(last_file).split('&')[0]
        last_file = os.path.join(download_dir, last_file)
    except:
        last_file = ['']

    while True:
        time.sleep(CHECK_INTERVAL)
        current_files = get_online_file_list(FOLDER_URL, options)
        current_files.sort()
            
        # Detect new files
        added_file = list(set(current_files) - set(previous_files))
        previous_files = current_files

        if added_file:
            print(f"New file detected: {added_file}")
            with open(".temp/current_files.txt", "r") as f:
                last_file = f.read().split("\n")[-1]
            last_file = urllib.parse.unquote(last_file) 
            last_file = os.path.basename(last_file).split('&')[0]
            last_file = os.path.join(download_dir, last_file)

            new_file = download_online_file(added_file[0], download_dir, options)

        with open(".temp/current_files.txt", "w") as f:
            f.write('\n'.join(str(i) for i in current_files))

        if added_file and last_file != download_dir:
            py_script = os.path.join(os.getcwd(), "src/aimon/main.py")
            cmd = f"python {py_script} -c {config_json_path} -f {last_file} {new_file}"
            print(f"Executing command line:\n{cmd}")
            subprocess.call(["python", py_script, "-c", config_json_path, "-f", last_file, new_file], cwd=os.getcwd())

            # Convert the change events GeoJSON file to a datamodel JSON file
            json_settings = read_json_file(config_json_path)
            output_folder = json_settings["project_setting"]["output_folder"]
            project_name = json_settings["project_setting"]["project_name"]

            geojson_name = os.path.join(os.getcwd(), output_folder, project_name, "04_Change_visualisation_UHD_Change_Events", "%s_change_events_pixel.geojson"%project_name)
            geojson_to_datamodel(geojson_name, datamodel_json_path)

            print("Waiting for another file...")
            
        #break

if __name__ == "__main__":
    FOLDER_URL = r'https://heibox.uni-heidelberg.de/d/8c6f08c92a1f48978fb4/files/' # URL to the server containing the files (3DGeo/3DGeo_Exchange/Will)
    CHECK_INTERVAL = 3  # seconds
    download_dir = r"/home/william/Downloads/" # Directory to download the files to

    config_json_path = r"/home/william/Documents/DATA/TRIER/project_settings_trier.json" # Path to the config json file
    datamodel_json_path = "/home/william/Documents/GitHub/4dgeo/public/testing_data/rockfall_monitoring_trier/sample_rockfall_monitoring_trier.json"

    # main(config_json_path, datamodel_json_path, download_dir)
    from aimon.helpers.utilities import upload_file
    options = webdriver.ChromeOptions()
    upload_file(r'https://heibox.uni-heidelberg.de/d/8c6f08c92a1f48978fb4/files/',
                "/media/william/f12874d3-cea9-494b-bae1-663bca9f9f86/TRIER/LowRes/240826_000005.laz",
                options)


