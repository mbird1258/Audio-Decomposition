import requests
from bs4 import BeautifulSoup
import urllib
import os

BaseDir = "https://theremin.music.uiowa.edu/"
Debug = True
urls = ["https://theremin.music.uiowa.edu/MISpiano.html", 
        "https://theremin.music.uiowa.edu/MISguitar.html", 
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISFlute2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISaltoflute2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBassFlute2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISOboe2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISEbClarinet2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBbClarinet2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBbBassClarinet2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBassoon2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISBbSopranoSaxophone2012.html",
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISEbAltoSaxophone2012.html",
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
        "https://theremin.music.uiowa.edu/MIS-Pitches-2012/MISTambourines2012.html"]

for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    for element in soup.find_all('p')[2:]:
        for child in element.findChildren("a", href = True):
            try:
                if child['href'] == "oops.html":
                    continue
                
                name = "/Users/matthewbird/Documents/Python Code/Song Decomposition/InstrumentAudioFiles/" + child.text.split(" ")[0]
                if name[-3:] == "zip":
                    continue

                if os.path.isfile(name):
                    if os.path.getsize(name):
                        if Debug: print(f"{name} already exists, skipping")
                        continue

                file = open(name, 'wb')
                file.write(urllib.request.urlopen(urllib.parse.urljoin(BaseDir, child['href'].replace(" ", "%20"))).read())
                if Debug: print(f"{name} successfully added")
                file.close()
            except Exception as e:
                if Debug: print(f"\n\n\n Exception encountered: {e} \n\n\n Skipped file: {name} \n\n\n")

print("\nfin")