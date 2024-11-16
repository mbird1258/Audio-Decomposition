import requests
from bs4 import BeautifulSoup
import urllib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

BaseDir = "https://theremin.music.uiowa.edu/"
Debug = True  # Set to True to enable debugging

urls = ["https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBbSopranoSaxophone2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISEbAltoSaxophone2012.html",
        "https://theremin.music.uiowa.edu/MISpiano.html",
        "https://theremin.music.uiowa.edu/MISguitar.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISFlute2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISaltoflute2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBassFlute2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISOboe2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISEbClarinet2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBbClarinet2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBbBassClarinet2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBassoon2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISHorn2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBbTrumpet2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISTenorTrombone2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBassTrombone2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISTuba2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISViolin2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISViola2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISCello2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISDoubleBass2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISMarimba2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISxylophone2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISVibraphone2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBells2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISCymbals2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISGongsTamTams2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISHandPercussion2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISTambourines2012.html"
]

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def download_file(child, processed_files, total_files, lock, url):
    try:
        if child['href'] == "oops.html":
            with lock:
                processed_files[0] += 1
                print_progress_bar(processed_files[0], total_files, prefix=f'Progress: {url}', suffix='Complete', length=50)
            return

        name = child.text.split(" ")[0].replace('\r', '').replace('\n', '').strip()
        name = os.path.join(os.path.dirname(os.path.realpath(__file__)), "InstrumentAudioFiles", name)

        if name[-3:] == "zip":
            with lock:
                processed_files[0] += 1
                print_progress_bar(processed_files[0], total_files, prefix=f'Progress: {url}', suffix='Complete', length=50)
            return

        if os.path.isfile(name):
            if os.path.getsize(name):
                if Debug: print(f"{name} already exists, skipping")
                with lock:
                    processed_files[0] += 1
                    print_progress_bar(processed_files[0], total_files, prefix=f'Progress: {url}', suffix='Complete', length=50)
                return
        elif os.path.isdir(name):
            with lock:
                processed_files[0] += 1
                print_progress_bar(processed_files[0], total_files, prefix=f'Progress: {url}', suffix='Complete', length=50)
            return

        with open(name, 'wb') as file:
            file.write(urllib.request.urlopen(urllib.parse.urljoin(BaseDir, child['href'].replace(" ", "%20"))).read())
            if Debug: print(f"{name} successfully added")
        with lock:
            processed_files[0] += 1
            print_progress_bar(processed_files[0], total_files, prefix=f'Progress: {url}', suffix='Complete', length=50)
    except Exception as e:
        if Debug: print(f"\n\n\n Exception encountered: {e} \n\n\n Skipped file: {name} \n\n\n")
        with lock:
            processed_files[0] += 1
            print_progress_bar(processed_files[0], total_files, prefix=f'Progress: {url}', suffix='Complete', length=50)

def process_url(url, processed_files, total_files, lock):
    if Debug: print(f"Processing URL: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    elements = soup.find_all('p')[2:]
    total_files_for_url = sum(len(element.findChildren("a", href=True)) for element in elements)

    with ThreadPoolExecutor() as executor:
        futures = []
        for element in elements:
            for child in element.findChildren("a", href=True):
                futures.append(executor.submit(download_file, child, processed_files, total_files, lock, url))
        for future in futures:
            future.result()

def fetch_url_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    elements = soup.find_all('p')[2:]
    return sum(len(element.findChildren("a", href=True)) for element in elements)

def main():
    if Debug: print("Calculating total number of files...")
    total_files = 0
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_url_content, url) for url in urls]
        for future in as_completed(futures):
            total_files += future.result()
    if Debug: print(f"Total number of files: {total_files}")
    processed_files = [0]
    lock = threading.Lock()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_url, url, processed_files, total_files, lock) for url in urls]
        for future in futures:
            future.result()

    print("\nAll downloads complete.")

if __name__ == '__main__':
    main()
