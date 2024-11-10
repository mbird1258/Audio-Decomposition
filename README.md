# Audio Decomposition
Blog post: [https://matthew-bird.com/blogs/Audio-Decomposition.html](https://matthew-bird.com/blogs/Audio-Decomposition.html)

### Youtube videos
[![Demo Video 1](https://img.youtube.com/vi/-i0PSxcoDH0/0.jpg)](https://www.youtube.com/watch?v=-i0PSxcoDH0)
[![Demo Video 2](https://img.youtube.com/vi/LkDJ9XT-klU/0.jpg)](https://www.youtube.com/watch?v=LkDJ9XT-klU)
[![Demo Video 3](https://img.youtube.com/vi/Z7D6obv12zk/0.jpg)](https://www.youtube.com/watch?v=Z7D6obv12zk)
[![Demo Video 4](https://img.youtube.com/vi/mzPUfs9sYQE/0.jpg)](https://www.youtube.com/watch?v=mzPUfs9sYQE)

## Premise
My plan for this project was to create a program to turn music to sheet music. It was mainly incentivised by my own desire to turn music to sheet music and the lack (from what I could tell) of open source, simple algorithms to perform audio source separation. 

## Preparation
### Instrument data
The instrument data all comes from the University of Iowa Electronic Music Studios instrument database. With these files, we find the Fourier transform of the entire wave and the envelope of the wave using the same method as documented below. 

## How it works
An instrument’s sound wave is mainly characterized by its fourier transform and envelope. Thus, we can use both of these to hopefully get a good idea of which instrument is playing what note. 

### Fourier Transform
The program’s first method of splitting music into constituent notes and instruments is by taking the fourier transform of the music file every 0.1 seconds (spectrogram), and adding together our stored fourier transform of each instrument to recreate the fourier transform of the 0.1 second window. The idea was to hopefully perfectly recreate the music at the set time as the fourier transform should represent the music played relatively well. 


<ins>Original fourier transform</ins>

![image](https://github.com/user-attachments/assets/ae2d9d74-7d5f-4422-bf8b-7cc9f8e7e5fe)


<ins>Constituent instruments</ins>

![image](https://github.com/user-attachments/assets/57d5e0ce-8245-4de3-8e0c-45e3f55ed961)


<ins>Recreated fourier transform</ins>

![image](https://github.com/user-attachments/assets/795b96de-92c4-44db-b830-2efd5db5f260)


The magnitudes for each instrument are given by solving the following matrix. The matrix is derived from taking the partial derivative of the MSE cost function by frequency(ex. FT value at 5 hz) with respect to each instrument. Each row in the matrix is a different partial derivative. (First is w.r.t cello, second is w.r.t piano, etc.)

<img width="519" alt="Screenshot 2024-10-05 at 8 32 50 PM" src="https://github.com/user-attachments/assets/4600c97f-ea83-4dd4-8977-2db3d9e703c1">

### Envelope
The first step to matching the envelope of the instrument to the sound wave is to obtain the envelope itself. The envelope is the upper bound of a wave, and although there are functions to do this, they seemingly struggle with noise and certain types of sound waves. Thus, since we have to handle many different instruments at different frequencies, we need a more robust solution. 

To get the envelope, the function splits the sound wave into chunks, before taking the max value at each chunk. To further refine the results, the function finds the points where the envelope is below the original sound wave and adds a new point defining the envelope. 

<img width="300" alt="IMG_0333" src="https://github.com/user-attachments/assets/d3219d4a-01de-4698-8539-b095c1e330ea">
<img width="300" alt="IMG_0334" src="https://github.com/user-attachments/assets/bbbf1bee-62b6-4ef3-8c78-b4f3bbbefbfe">
<img width="300" alt="IMG_0335" src="https://github.com/user-attachments/assets/d6c1734a-6a58-4759-bab7-3dcf169dc997">
<img width="300" alt="IMG_0336" src="https://github.com/user-attachments/assets/f0549389-f4c2-4672-836b-65be877cf581">


The next step is to split the envelope of the wave into its attack, sustain, and release. The attack is the initial noise of the note, the sustain is while the note is held, and the release is when the note stops. For the instrument samples, we can take the first nonzero value of the wave to get the start of the attack. To get the point between the attack and sustain, we get the first point when the function is concave down or decreasing. To get the point between the sustain and release, we get the first point from the end where the function is increasing or concave down. To get the end of the release, we find the first point from the end where the function is nonzero. 

To further classify the wave, we need to take into account the main forms the wave can take. Some instruments, such as the piano, have static decay, in which they mostly follow an exponential decay shape. On the other hand, some instruments, like the violin, can increase or decrease in volume as the note is sustained. In addition to this, some audio samples in the instrument files are held until their sound expires, while others are released early. To differentiate whether the decay is static or dynamic, if the decay factor is >1, or if it deviates from the decay curve by too much, it’s dynamic. To differentiate if the envelope has a release or not(AS or ASR), we look at the average rate of change across the sustain and the release, and if the rate of change of the release is lower then there is no release. 

<img width="519" alt="IMG_0331" src="https://github.com/user-attachments/assets/5953d36f-b56c-474d-bd9f-12aaefceffb5">

To deal with the music file, we first take the bandpass filter of the signal for each note frequency. With the filtered wave, we iterate through each instrument. For each instrument, we take the cross correlation of the instrument’s attack(normalized) and release(normalized) to find the start and end of each note, and then take the MSE of the instrument wave and the filtered audio to get our cost for the instrument at that time. After this, we multiply the magnitude we found in the fourier transform step and 1/(cost we found in this step) to get our final magnitudes. 

### Display
To display the file, we use matplotlib’s scatter plot with - shaped points to display the sheet music. Originally, I wanted to recreate the audio from the magnitudes, but it led to many issues, took a while, and made troubleshooting much harder. I also tried using matplotlib’s imshow plot, but it’s extremely inefficient in this case as most values are 0, and matplotlib needs to redraw every point regardless of if it’s on screen or not every time we pan or zoom the screen. 

![image](https://github.com/user-attachments/assets/af897aca-0497-463f-82b8-f32d81394932)


## Results
Overall, I think it works quite well. You can use it to make recreating sheet music better(as I did here from this video), especially if you struggle with finding the right pitch or chords, and it doesn’t take too much time to run either. 

## How to run project
1. (Only needs to be run once after downloading from GitHub) run ScrapeInstruments.py and ProcessInstruments.py one time each (will take a while)
   InstrumentAudioFiles and InstrumentData should now be filled
3. Upload filetypes that soundfile.read() can process to the In folder
4. Go into Main.py and change any parameters, primarily the whitelist or blacklist of instruments for the song
5. Run Main.py
   PlayBack should now have a file for each input
7. Run Display.py to see results!
