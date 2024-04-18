 #!/usr/bin/env python
# author: rmarkello
# created: 2015-12-07

from __future__ import division
import os
from psychopy import visual, event, core, gui
import numpy as np

if __name__ == "__main__":
    expInfo = {'modality':('EKG','PPG','RESP')}
   
    dlg = gui.DlgFromDict(dictionary=expInfo,
                      title='Wave Display',
                      order=['modality'])
    if not dlg.OK: core.quit()
   
    if expInfo['modality'] == 'EKG':
        CHANNEL        =   9
        MULTIPLIER     =   3
        SAMPLE_RATE    =   1000

    elif expInfo['modality'] == 'PPG':
        CHANNEL        =   2
        MULTIPLIER     =   2
        SAMPLE_RATE    =   200

    elif expInfo['modality'] == 'RESP':
        CHANNEL        =   1
        MULTIPLIER     =   10
        SAMPLE_RATE    =   200

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    from rtpeaks import MP150 as biopac
    print("imported rtpeaks as biopac")
    
    # display
    win = visual.Window([1920,1080],fullscr=True,units='norm',color='gray')
    win.setMouseVisible(False)

    cancel = visual.TextStim(win,text='Press escape to exit', pos=(0,-0.75))
    cancel.setAutoDraw(True)
    print("creating waveform")
    # shape for waveform (starts at center+ of screen)
    wave_form = visual.ShapeStim(win,closeShape=False,vertices=[[0,0],[0,0]])
    wave_form.setAutoDraw(True)
    print("creating output file")
    # start communication with MP150
    mp = biopac(logfile='dummy', samplerate=SAMPLE_RATE, channels=[CHANNEL])
    print("line 48 finished")
    win.winHandle.activate()
    print("window activated")

    # display a constantly updating waveform on the screen
    # press escape to exit
    while not len(event.getKeys(keyList=['escape'])):
        # grab a sample from MP 150
        sample = mp.sample[0]/MULTIPLIER
        curr_point = (0,sample)
        num_vert = wave_form.vertices.shape[0]
        print("sample acquired")
        # move the waveform along the screen until it extends to the left edge
        if num_vert < 100:
            times, points = np.split(wave_form.vertices,2,axis=1)
            times = (np.arange(-1*num_vert,1)/100).reshape(num_vert+1,1)
            points = np.vstack((points,curr_point[1]))
            wave_form.vertices = np.hstack((times,points))
        else:
            times, points = np.split(wave_form.vertices,2,axis=1)
            points = np.vstack((np.split(points,[1],axis=0)[1],curr_point[1]))
            wave_form.vertices = np.hstack((times,points))

        # refresh screen
        win.flip()

    mp.close()
    win.close()
