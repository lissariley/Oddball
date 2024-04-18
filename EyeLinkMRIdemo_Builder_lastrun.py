#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on March 07, 2024, at 09:32
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from elConnect
# DESCRIPTION:
# This is a basic example illustrating how to do continuous eye tracker 
# recording through a block of trials (e.g., in an MRI setup), and how to 
# synchronize the presentation of trials with a sync signal from the MRI. With 
# a long recording, we start and stop recording at the beginning and end of a 
# testing session (block/run), rather than at the beginning and end of each 
# experimental trial. We still send the TRIALID and TRIAL_RESULT messages to 
# the tracker, and Data Viewer will still be able to segment the long recording 
# into small segments (trials).

# The code components in the eyelinkSetup, eyelinkStartRecording, trial, and 
# eyelinkStopRecording routines handle communication with the Host PC/EyeLink
# system.  All the code components are set to Code Type Py, and each code 
# component may have code in the various tabs (e.g., Before Experiment, Begin
# Experiment, etc.)

# Last updated: October 27 2023

# This Before Experiment tab of the elConnect component imports some
# modules we need, manages data filenames, allows for dummy mode configuration
# (for testing), connects to the Host PC, and defines some helper function 
# definitions (which are called later)

import pylink
import time
import platform
from PIL import Image  # for preparing the Host backdrop image
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from string import ascii_letters, digits
from psychopy import gui

# Switch to the script folder
script_path = os.path.dirname(sys.argv[0])
if len(script_path) != 0:
    os.chdir(script_path)

# Set this variable to True if you use the built-in retina screen as your
# primary display device on macOS. If have an external monitor, set this
# variable True if you choose to "Optimize for Built-in Retina Display"
# in the Displays preference settings.
use_retina = False

# Set this variable to True to run the script in "Dummy Mode"
dummy_mode = False

# Set up EDF data file name and local data folder
#
# The EDF data filename should not exceed 8 alphanumeric characters
# use ONLY number 0-9, letters, & _ (underscore) in the filename
edf_fname = 'TEST'

# Prompt user to specify an EDF data filename
# before we open a fullscreen window
dlg_title = 'Enter EDF File Name'
dlg_prompt = 'Please enter a file name with 8 or fewer characters\n' + \
             '[letters, numbers, and underscore].'
# loop until we get a valid filename
while True:
    dlg = gui.Dlg(dlg_title)
    dlg.addText(dlg_prompt)
    dlg.addField('File Name:', edf_fname)
    # show dialog and wait for OK or Cancel
    ok_data = dlg.show()
    if dlg.OK:  # if ok_data is not None
        print('EDF data filename: {}'.format(ok_data[0]))
    else:
        print('user cancelled')
        core.quit()
        sys.exit()

    # get the string entered by the experimenter
    tmp_str = dlg.data[0]
    # strip trailing characters, ignore the ".edf" extension
    edf_fname = tmp_str.rstrip().split('.')[0]

    # check if the filename is valid (length <= 8 & no special char)
    allowed_char = ascii_letters + digits + '_'
    if not all([c in allowed_char for c in edf_fname]):
        print('ERROR: Invalid EDF filename')
    elif len(edf_fname) > 8:
        print('ERROR: EDF filename should not exceed 8 characters')
    else:
        break
        
# Set up a folder to store the EDF data files and the associated resources
# e.g., files defining the interest areas used in each trial
results_folder = 'results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# We download EDF data file from the EyeLink Host PC to the local hard
# drive at the end of each testing session, here we rename the EDF to
# include session start date/time
time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
session_identifier = edf_fname + time_str

# create a folder for the current testing session in the "results" folder
session_folder = os.path.join(results_folder, session_identifier)
if not os.path.exists(session_folder):
    os.makedirs(session_folder)

# For macOS users check if they have a retina screen
if 'Darwin' in platform.system():
        dlg = gui.Dlg("Retina Screen?")
        dlg.addText("Will the task run on a Retina or a non-Retina screen?")
        dlg.addField("Screen Type:", choices=["Not Retina","Retina"])
        # show dialog and wait for OK or Cancel
        ok_data = dlg.show()
        if dlg.OK:
            if dlg.data[0] == "Retina":  
                use_retina = True
            else:
                use_retina = False
        else:
            print('user cancelled')
            core.quit()
            sys.exit()


# Step 1: Connect to the EyeLink Host PC
#
# The Host IP address, by default, is "100.1.1.1".
# the "el_tracker" objected created here can be accessed through the Pylink
# Set the Host PC address to "None" (without quotes) to run the script
# in "Dummy Mode"
if dummy_mode:
    el_tracker = pylink.EyeLink(None)
else:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        dlg = gui.Dlg("Dummy Mode?")
        dlg.addText("Couldn't connect to tracker at 100.1.1.1 -- continue in Dummy Mode?")
        # show dialog and wait for OK or Cancel
        ok_data = dlg.show()
        if dlg.OK:  # if ok_data is not None
            dummy_mode = True
            el_tracker = pylink.EyeLink(None)
        else:
            print('user cancelled')
            core.quit()
            sys.exit()

# Define some helper functions for screen drawing 
# and exiting trials/sessions early
def clear_screen(win,genv):
    """ clear up the PsychoPy window"""
    win.fillColor = genv.getBackgroundColor()
    win.flip()

def show_msg(win, genv, text, wait_for_keypress=True):
    """ Show task instructions on screen"""
    scn_width, scn_height = win.size
    msg = visual.TextStim(win, text,
                          color=genv.getForegroundColor(),
                          wrapWidth=scn_width/2)
    clear_screen(win,genv)
    msg.draw()
    win.flip()

    # wait indefinitely, terminates upon any key press
    if wait_for_keypress:
        kb = keyboard.Keyboard()
        #keys = kb.getKeys(['Enter'], waitRelease=False)
        waitKeys = kb.waitKeys(keyList=None, waitRelease=True, clear=True)
        clear_screen(win,genv)

def terminate_task(genv,edf_file,session_folder,session_identifier):
    """ Terminate the task gracefully and retrieve the EDF data file
    """
    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected():
        # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial()

        # Put tracker in Offline mode
        el_tracker.setOfflineMode()

        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the edf data file on the Host
        el_tracker.closeDataFile()

        # Show a file transfer message on the screen
        msg = 'EDF data is transferring from EyeLink Host PC...'
        show_msg(win, genv, msg, wait_for_keypress=False)

        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        local_edf = os.path.join(session_folder, session_identifier + '.EDF')
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()

    # close the PsychoPy window
    win.close()

    # quit PsychoPy
    core.quit()
    sys.exit()

def abort_trial():
    """Ends recording """
    el_tracker = pylink.getEYELINK()

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()
        
    # Send a message to clear the Data Viewer screen
    bgcolor_RGB = (128, 128, 128)
    el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)

    return pylink.TRIAL_ERROR

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'oddball_with_tracking'  # from the Builder filename that created this script
expInfo = {
    'participant': '999',
    'age': '38',
    'tracking': '0',
    'scanning': '0',
    'practice': '0',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data' + os.sep + '%s_%s' % (expInfo['participant'], expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\aclab\\Desktop\\Oddball\\oddball_buildup\\EyeLinkMRIdemo_Builder_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.DEBUG)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.DEBUG)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1920, 1080], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color='[0.3082,0.3536,0.2946]', colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='pix'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = '[0.3082,0.3536,0.2946]'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'pix'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='ptb')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='PsychToolbox')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    welcometext = visual.TextStim(win=win, name='welcometext',
        text='Welcome to\nthe Oddball Task!\nRemember, your job is to \npress the button under\nyour first finger when\nyou see the LARGE circle.',
        font='Open Sans',
        pos=(0, 0), height=50.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    move_on_from_welcome = keyboard.Keyboard()
    
    # --- Initialize components for Routine "eyelinkSetup" ---
    elInstructions = visual.TextStim(win=win, name='elInstructions',
        text='If using eyetracking, press any key to enter camera setup',
        font='Open Sans',
        pos=(0, 0), height=50.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    # Run 'Begin Experiment' code from elConnect
    # This Begin Experiment tab of the elConnect component opens the EDF, gets graphic 
    # information from Psychopy, configures some eye tracker settings, logs the screen 
    # resolution for Data Viewer via a DISPLAY_COORDS message, and configures a 
    # graphics environment for eye tracker setup/calibration
    if int(expInfo['tracking'])==0:
        print('*************************************************skipping elConnect')
        continueRoutine = False
    elif int(expInfo['tracking'])==1:
        print('*************************************************proceeding with elConnect')
    
    el_tracker = pylink.getEYELINK()
    # Step 2: Open an EDF data file on the Host PC
    global edf_fname
    edf_file = edf_fname + ".EDF"
    try:
        el_tracker.openDataFile(edf_file)
    except RuntimeError as err:
        print('ERROR:', err)
        # close the link if we have one open
        if el_tracker.isConnected():
            el_tracker.close()
        core.quit()
        sys.exit()
    
    # Add a header text to the EDF file to identify the current experiment name
    # This is OPTIONAL. If your text starts with "RECORDED BY " it will be
    # available in DataViewer's Inspector window by clicking
    # the EDF session node in the top panel and looking for the "Recorded By:"
    # field in the bottom panel of the Inspector.
    preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
    el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)
    
    # Step 3: Configure the tracker
    #
    # Put the tracker in offline mode before we change tracking parameters
    el_tracker.setOfflineMode()
    
    # Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
    # 5-EyeLink 1000 Plus, 6-Portable DUO
    eyelink_ver = 0  # set version to 0, in case running in Dummy mode
    if not dummy_mode:
        vstr = el_tracker.getTrackerVersionString()
        eyelink_ver = int(vstr.split()[-1].split('.')[0])
        # print out some version info in the shell
        print('Running experiment on %s, version %d' % (vstr, eyelink_ver))
    
    # File and Link data control
    # what eye events to save in the EDF file, include everything by default
    file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
    # what eye events to make available over the link, include everything by default
    link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
    # what sample data to save in the EDF data file and to make available
    # over the link, include the 'HTARGET' flag to save head target sticker
    # data for supported eye trackers
    if eyelink_ver > 3:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
    else:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,PUPIL,AREA,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
    el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
    el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
    el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
    el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)
    
    # Optional tracking parameters
    # Sample rate, 250, 500, 1000, or 2000, check your tracker specification
    # if eyelink_ver > 2:
    #     el_tracker.sendCommand("sample_rate 1000")
    # Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
    el_tracker.sendCommand("calibration_type = HV9")
    # Set a gamepad button to accept calibration/drift check target
    # You need a supported gamepad/button box that is connected to the Host PC
    el_tracker.sendCommand("button_function 5 'accept_target_fixation'")
    
    # get the native screen resolution used by PsychoPy
    scn_width, scn_height = win.size
    # resolution fix for Mac retina displays
    if 'Darwin' in platform.system():
        if use_retina:
            scn_width = int(scn_width/2.0)
            scn_height = int(scn_height/2.0)
    
    # Pass the display pixel coordinates (left, top, right, bottom) to the tracker
    # see the EyeLink Installation Guide, "Customizing Screen Settings"
    el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendCommand(el_coords)
    
    # Write a DISPLAY_COORDS message to the EDF file
    # Data Viewer needs this piece of info for proper visualization, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendMessage(dv_coords)  
        
    # Configure a graphics environment (genv) for tracker calibration
    genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
    print(genv)  # print out the version number of the CoreGraphics library
    
    
    # --- Initialize components for Routine "instruct" ---
    taskInstructions = visual.TextStim(win=win, name='taskInstructions',
        text='Ready?\n\nPress space to continue.',
        font='Open Sans',
        pos=(0, 0), height=50.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    ready = keyboard.Keyboard()
    
    # --- Initialize components for Routine "eyelinkStartRecording" ---
    # Run 'Begin Experiment' code from elStartRecord
    # This Begin Experiment tab of the elStartRecord component initializes some 
    # variables that are used to keep track of the current trial
    # numbers
    
    trial_index = 1
    
    # --- Initialize components for Routine "waitForScannerPulse" ---
    waitTriggerText = visual.TextStim(win=win, name='waitTriggerText',
        text='Waiting for scanner...',
        font='Open Sans',
        pos=(0, 0), height=50.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    keyTrigger = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial" ---
    resp = keyboard.Keyboard()
    fixation = visual.ShapeStim(
        win=win, name='fixation',units='deg', 
        size=(0.15, 0.15), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=-2.0, interpolate=True)
    ball = visual.Polygon(
        win=win, name='ball',units='deg', 
        edges=200, size=[1.0, 1.0],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "eyelinkStopRecording" ---
    
    # --- Initialize components for Routine "thanks" ---
    endScreen = visual.TextStim(win=win, name='endScreen',
        text='This is the end of the experiment.\n\nThanks!',
        font='Open Sans',
        pos=(0, 0), height=50.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('welcome.started', globalClock.getTime())
    move_on_from_welcome.keys = []
    move_on_from_welcome.rt = []
    _move_on_from_welcome_allKeys = []
    # keep track of which components have finished
    welcomeComponents = [welcometext, move_on_from_welcome]
    for thisComponent in welcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcometext* updates
        
        # if welcometext is starting this frame...
        if welcometext.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcometext.frameNStart = frameN  # exact frame index
            welcometext.tStart = t  # local t and not account for scr refresh
            welcometext.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcometext, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcometext.started')
            # update status
            welcometext.status = STARTED
            welcometext.setAutoDraw(True)
        
        # if welcometext is active this frame...
        if welcometext.status == STARTED:
            # update params
            pass
        
        # *move_on_from_welcome* updates
        waitOnFlip = False
        
        # if move_on_from_welcome is starting this frame...
        if move_on_from_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            move_on_from_welcome.frameNStart = frameN  # exact frame index
            move_on_from_welcome.tStart = t  # local t and not account for scr refresh
            move_on_from_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(move_on_from_welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'move_on_from_welcome.started')
            # update status
            move_on_from_welcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(move_on_from_welcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(move_on_from_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if move_on_from_welcome.status == STARTED and not waitOnFlip:
            theseKeys = move_on_from_welcome.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _move_on_from_welcome_allKeys.extend(theseKeys)
            if len(_move_on_from_welcome_allKeys):
                move_on_from_welcome.keys = _move_on_from_welcome_allKeys[-1].name  # just the last key pressed
                move_on_from_welcome.rt = _move_on_from_welcome_allKeys[-1].rt
                move_on_from_welcome.duration = _move_on_from_welcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('welcome.stopped', globalClock.getTime())
    # check responses
    if move_on_from_welcome.keys in ['', [], None]:  # No response was made
        move_on_from_welcome.keys = None
    thisExp.addData('move_on_from_welcome.keys',move_on_from_welcome.keys)
    if move_on_from_welcome.keys != None:  # we had a response
        thisExp.addData('move_on_from_welcome.rt', move_on_from_welcome.rt)
        thisExp.addData('move_on_from_welcome.duration', move_on_from_welcome.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "eyelinkSetup" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('eyelinkSetup.started', globalClock.getTime())
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    eyelinkSetupComponents = [elInstructions, key_resp]
    for thisComponent in eyelinkSetupComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "eyelinkSetup" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *elInstructions* updates
        
        # if elInstructions is starting this frame...
        if elInstructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            elInstructions.frameNStart = frameN  # exact frame index
            elInstructions.tStart = t  # local t and not account for scr refresh
            elInstructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(elInstructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'elInstructions.started')
            # update status
            elInstructions.status = STARTED
            elInstructions.setAutoDraw(True)
        
        # if elInstructions is active this frame...
        if elInstructions.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in eyelinkSetupComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "eyelinkSetup" ---
    for thisComponent in eyelinkSetupComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('eyelinkSetup.stopped', globalClock.getTime())
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # Run 'End Routine' code from elConnect
    if int(expInfo['tracking'])==1:
    
    # This End Routine tab of the elConnect component configures some
    # graphics options for calibration, and then performs a camera setup
    # so that you can set up the eye tracker and calibrate/validate the participant
    
    # Set background and foreground colors for the calibration target
    # in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
        foreground_color = (-1, -1, -1)
        background_color = tuple(win.color)
        genv.setCalibrationColors(foreground_color, background_color)
    
        print('Got here: A')
    
    # Set up the calibration target
    #
    # The target could be a "circle" (default), a "picture", a "movie" clip,
    # or a rotating "spiral". To configure the type of calibration target, set
    # genv.setTargetType to "circle", "picture", "movie", or "spiral", e.g.,
    # genv.setTargetType('picture')
    #
    # Use genv.setMovieTarget() to set a "movie" target
    # genv.setMovieTarget(os.path.join('videos', 'calibVid.mov'))
    
    # Use a picture as the calibration target
        genv.setTargetType('picture')
        genv.setPictureTarget(os.path.join('images', 'fixTarget.bmp'))
    
        print('Got here: B')
    
    # Configure the size of the calibration target (in pixels)
    # this option applies only to "circle" and "spiral" targets
    # genv.setTargetSize(24)
    
    # Beeps to play during calibration, validation and drift correction
    # parameters: target, good, error
    #     target -- sound to play when target moves
    #     good -- sound to play on successful operation
    #     error -- sound to play on failure or interruption
    # Each parameter could be ''--default sound, 'off'--no sound, or a wav file
        genv.setCalibrationSounds('', '', '')
    
    # resolution fix for macOS retina display issues
        if use_retina:
            genv.fixMacRetinaDisplay()
    
    #clear the screen before we begin Camera Setup mode
        clear_screen(win,genv)
    
        print('Got here: C')
    
    # Request Pylink to use the PsychoPy window we opened above for calibration
        pylink.openGraphicsEx(genv)
    
        print('Got here: D')
    
    # Peform a Camera Setup (eye tracker calibration)
    # skip this step if running the script in Dummy Mode
        if not dummy_mode:
            try:
                el_tracker.doTrackerSetup()
            except RuntimeError as err:
                print('ERROR:', err)
                el_tracker.exitCalibration()
        clear_screen(win,genv)
    
        print('Got here: E')
    # the Routine "eyelinkSetup" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruct.started', globalClock.getTime())
    ready.keys = []
    ready.rt = []
    _ready_allKeys = []
    # keep track of which components have finished
    instructComponents = [taskInstructions, ready]
    for thisComponent in instructComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *taskInstructions* updates
        
        # if taskInstructions is starting this frame...
        if taskInstructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            taskInstructions.frameNStart = frameN  # exact frame index
            taskInstructions.tStart = t  # local t and not account for scr refresh
            taskInstructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(taskInstructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'taskInstructions.started')
            # update status
            taskInstructions.status = STARTED
            taskInstructions.setAutoDraw(True)
        
        # if taskInstructions is active this frame...
        if taskInstructions.status == STARTED:
            # update params
            pass
        
        # *ready* updates
        waitOnFlip = False
        
        # if ready is starting this frame...
        if ready.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            ready.frameNStart = frameN  # exact frame index
            ready.tStart = t  # local t and not account for scr refresh
            ready.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ready, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'ready.started')
            # update status
            ready.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(ready.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(ready.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if ready.status == STARTED and not waitOnFlip:
            theseKeys = ready.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _ready_allKeys.extend(theseKeys)
            if len(_ready_allKeys):
                ready.keys = _ready_allKeys[-1].name  # just the last key pressed
                ready.rt = _ready_allKeys[-1].rt
                ready.duration = _ready_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct" ---
    for thisComponent in instructComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruct.stopped', globalClock.getTime())
    # the Routine "instruct" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "eyelinkStartRecording" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('eyelinkStartRecording.started', globalClock.getTime())
    # Run 'Begin Routine' code from elStartRecord
    if int(expInfo['tracking'])==1:
        print('*************************************************proceeding with elStartRecord')
    
    # This Begin Routine tab of the elStartRecord component updates some 
    # variables that are used to keep track of the current trial and block 
    # numbers, draws some feedback graphics (a simple shape) on the 
    # Host PC, sends a trial start messages to the EDF, performs a 
    # drift check/drift correct, and starts eye tracker recording
    
    # get a reference to the currently active EyeLink connection
        el_tracker = pylink.getEYELINK()
    
    # put the tracker in the offline mode first
        el_tracker.setOfflineMode()
    
    # clear the host screen before we draw the backdrop
        el_tracker.sendCommand('clear_screen 0')
    
    # OPTIONAL: draw landmarks and texts on the Host screen
    # In addition to backdrop image, You may draw simples on the Host PC to use
    # as landmarks. For illustration purpose, here we draw some texts and a box
    # For a list of supported draw commands, see the "COMMANDS.INI" file on the
    # Host PC (under /elcl/exe)
        left = int(scn_width/2.0) - 60
        top = int(scn_height/2.0) - 60
        right = int(scn_width/2.0) + 60
        bottom = int(scn_height/2.0) + 60
        draw_cmd = 'draw_filled_box %d %d %d %d 1' % (left, top, right, bottom)
        el_tracker.sendCommand(draw_cmd)
    
    # send a "TRIALID" message to mark the start of a trial, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
        el_tracker.sendMessage('TRIALID %d' % trial_index)
    
    # drift check
    # we recommend drift-check at the beginning of each trial
    # the doDriftCorrect() function requires target position in integers
    # the last two arguments:
    # draw_target (1-default, 0-draw the target then call doDriftCorrect)
    # allow_setup (1-press ESCAPE to recalibrate, 0-not allowed)
    
    # Skip drift-check if running the script in Dummy Mode
        while not dummy_mode:
        # terminate the task if no longer connected to the tracker or
        # user pressed Ctrl-C to terminate the task
            if (not el_tracker.isConnected()) or el_tracker.breakPressed():
                terminate_task(genv,edf_file,session_folder,session_identifier)
        # drift-check and re-do camera setup if ESCAPE is pressed
            try:
                error = el_tracker.doDriftCorrect(int(scn_width/2.0),
                                              int(scn_height/2.0), 1, 1)
            # break following a success drift-check
                if error is not pylink.ESC_KEY:
                    break
            except:
                pass
    
    # put tracker in idle/offline mode before recording
        el_tracker.setOfflineMode()
    
    # Start recording
    # arguments: sample_to_file, events_to_file, sample_over_link,
    # event_over_link (1-yes, 0-no)
        try:
            el_tracker.startRecording(1, 1, 1, 1)
        except RuntimeError as error:
            print("ERROR:", error)
            abort_trial()
    
    # Allocate some time for the tracker to cache some samples before allowing
    # trial stimulus presentation to proceed
        pylink.pumpDelay(100)
    
    # keep track of which components have finished
    eyelinkStartRecordingComponents = []
    for thisComponent in eyelinkStartRecordingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "eyelinkStartRecording" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in eyelinkStartRecordingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "eyelinkStartRecording" ---
    for thisComponent in eyelinkStartRecordingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('eyelinkStartRecording.stopped', globalClock.getTime())
    # the Routine "eyelinkStartRecording" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "waitForScannerPulse" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('waitForScannerPulse.started', globalClock.getTime())
    # skip this Routine if its 'Skip if' condition is True
    continueRoutine = continueRoutine and not (scanning==0)
    keyTrigger.keys = []
    keyTrigger.rt = []
    _keyTrigger_allKeys = []
    # keep track of which components have finished
    waitForScannerPulseComponents = [waitTriggerText, keyTrigger]
    for thisComponent in waitForScannerPulseComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "waitForScannerPulse" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *waitTriggerText* updates
        
        # if waitTriggerText is starting this frame...
        if waitTriggerText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            waitTriggerText.frameNStart = frameN  # exact frame index
            waitTriggerText.tStart = t  # local t and not account for scr refresh
            waitTriggerText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(waitTriggerText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'waitTriggerText.started')
            # update status
            waitTriggerText.status = STARTED
            waitTriggerText.setAutoDraw(True)
        
        # if waitTriggerText is active this frame...
        if waitTriggerText.status == STARTED:
            # update params
            pass
        
        # *keyTrigger* updates
        waitOnFlip = False
        
        # if keyTrigger is starting this frame...
        if keyTrigger.status == NOT_STARTED and tThisFlip >= 0.25-frameTolerance:
            # keep track of start time/frame for later
            keyTrigger.frameNStart = frameN  # exact frame index
            keyTrigger.tStart = t  # local t and not account for scr refresh
            keyTrigger.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(keyTrigger, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'keyTrigger.started')
            # update status
            keyTrigger.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(keyTrigger.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(keyTrigger.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if keyTrigger.status == STARTED and not waitOnFlip:
            theseKeys = keyTrigger.getKeys(keyList=['5'], ignoreKeys=["escape"], waitRelease=False)
            _keyTrigger_allKeys.extend(theseKeys)
            if len(_keyTrigger_allKeys):
                keyTrigger.keys = _keyTrigger_allKeys[-1].name  # just the last key pressed
                keyTrigger.rt = _keyTrigger_allKeys[-1].rt
                keyTrigger.duration = _keyTrigger_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in waitForScannerPulseComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "waitForScannerPulse" ---
    for thisComponent in waitForScannerPulseComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('waitForScannerPulse.stopped', globalClock.getTime())
    # check responses
    if keyTrigger.keys in ['', [], None]:  # No response was made
        keyTrigger.keys = None
    thisExp.addData('keyTrigger.keys',keyTrigger.keys)
    if keyTrigger.keys != None:  # we had a response
        thisExp.addData('keyTrigger.rt', keyTrigger.rt)
        thisExp.addData('keyTrigger.duration', keyTrigger.duration)
    thisExp.nextEntry()
    # Run 'End Routine' code from logTriggerTime
    if expInfo['tracking']==1:
    
    # This End Routine tab of the logTriggerTime component sends an event
    # marking message for the trigger pulse signal and, importantly, logs the pulse
    # time so that we can later write make it a trial variable (via TRIAL_VAR
    # messages to the EDF on each trial)
    
    # If a key was presssed, calculate the difference between the current time 
    # and the time of the key press onset. This offset value will be sent at the 
    # beginning of the message and will automatically be subtracted by Data Viewer 
    # from the timestamp of the message to position the message at the correct point 
    # in time. Then send a message marking the event
        if not isinstance(keyTrigger.rt,list):
        
            offsetValue = int(round((globalClock.getTime() - \
                (keyTrigger.tStartRefresh + keyTrigger.rt))*1000))
            scanPulseTime = int(round((keyTrigger.tStartRefresh + keyTrigger.rt)*1000))
            el_tracker.sendMessage('%i SCAN_PULSE_RECEIVED' % offsetValue)
    # the Routine "waitForScannerPulse" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions.csv'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime())
        # Run 'Begin Routine' code from elTrial
        if int(expInfo['tracking'])==1:
        # This Begin Routine tab of the elTrial component resets some 
        # variables that are used to keep track of whether certain trial events have
        # happened and sends trial variable messages to the EDF to mark condition
        # information
        
        # these variables keep track of whether the fixation presentation, image 
        # presentation, and trial response have occured yet (0 = no, 1 = yes).
        # They later help us to ensure that each event marking message only gets
        # sent once, at the time of each event
            sentResponseMessage = 0
            sentFixationMessage = 0
            sentBallMessage = 0
        
        # create a keyboard instance and reinitialize a kePressNameList, which 
        # will store list of key names currently being pressed (to allow Ctrl-C abort)
            kb = keyboard.Keyboard()
            keyPressNameList = list()
        
        # send a "TRIALID" message to mark the start of a trial, see Data
        # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
        # Skip this message for the first trial
            if trial_index > 1:
                el_tracker.sendMessage('TRIALID %d' % trial_index)
            
        # record trial variables to the EDF data file, for details, see Data
        # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
            el_tracker.sendMessage('!V TRIAL_VAR stim_type %s' % stim_type)
            el_tracker.sendMessage('!V TRIAL_VAR stim_name %s' % stim_name)
            el_tracker.sendMessage('!V TRIAL_VAR stim_size %s' % stim_size)
            el_tracker.sendMessage('!V TRIAL_VAR RGB %s' % RGB)
        
        # if sending many messages in a row, add a 1 msec pause between after 
        # every 5 messages or so
            time.sleep(0.001)
            el_tracker.sendMessage('!V TRIAL_VAR corrAns %s' % corrAns)
        resp.keys = []
        resp.rt = []
        _resp_allKeys = []
        ball.setFillColor(RGB)
        ball.setSize(stim_size)
        ball.setLineColor(RGB)
        # keep track of which components have finished
        trialComponents = [resp, fixation, ball]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from elTrial
            if int(expInfo['tracking'])==1:
            
            # This Each Frame tab of the elTrial component handles the marking
            # of experimental events via messages to the EDF, sends additional messages
            # to allow visualization of trial stimuli in Data Viewer, logs trial variable
            # information associated with responses and stimulus timing, and checks whether
            # the eye tracker is still properly recording (and aborts the trial if not)
            
            # if a key was presssed, calculate the difference between the current time 
            # and the time of the key press onset. 
            # This offset value will be sent at the beginning of the message
            # and will automatically be subtracted by Data Viewer from the timestamp
            # of the message to position the message at the correct point in time
            # then send a message marking the event
                if not isinstance(resp.rt,list) and sentResponseMessage == 0:
                    offsetValue = int(round((globalClock.getTime() - \
                        (resp.tStartRefresh + resp.rt))*1000))
                    el_tracker.sendMessage('%i KEY_PRESSED' % offsetValue)
            
                # after every few messages, include a 1 msec delay to ensure
                # that no messages are missed 
                    time.sleep(0.001)
                    el_tracker.sendMessage('!V TRIAL_VAR accuracy %i' % resp.corr)
                    el_tracker.sendMessage('!V TRIAL_VAR keyPressed %s' % resp.keys)
            
                # log that the response message has been written so that we don't 
                # write it again on future frames
                    sentResponseMessage = 1
            
            #Check whether it is the first frame of the fixation presentation
                if fixation.tStartRefresh is not None and sentFixationMessage == 0:
                
                # send a message marking the fixation onset event
                    el_tracker.sendMessage('FIXATION_ONSET')
                
                # send some Data Viewer drawing commands so that you can see a representation
                # of the fixation cross in Data Viewer's various visualizations
                # For more information on this, see section "Protocol for EyeLink Data to 
                # Viewer Integration" section of the Data Viewer User Manual (Help -> Contents)
                    el_tracker.sendMessage('!V CLEAR 128 128 128')
                    el_tracker.sendMessage('!V DRAWLINE 255 255 255 %i %i %i %i' % \
                        (scn_width/2 - 25,scn_height/2,scn_width/2 + 25,\
                        scn_height/2))  
                    el_tracker.sendMessage('!V DRAWLINE 255 255 255 %i %i %i %i' % \
                        (scn_width/2,scn_height/2 - 25,scn_width/2,\
                        scn_height/2 + 25)) 
                    
                # log the fixation onset time (in Display PC time) as a Trial Variable
                    fixationTime = fixation.tStartRefresh*1000
                    el_tracker.sendMessage('!V TRIAL_VAR fixationTime %i' % fixationTime)
                
                # set this variable to 1 to ensure we don't write the event message/
                # draw command messages again on future frames
                    sentFixationMessage = 1
            
            # Check whether it is the first frame of the image presentation
                if ball.tStartRefresh is not None and sentBallMessage == 0:
            
                # send a message marking the image onset event
                    el_tracker.sendMessage('IMAGE_ONSET')  
            
                # send some Data Viewer drawing commands so that you can see the trial image
                # in Data Viewer's various visualizations
                # For more information on this, see section "Protocol for EyeLink Data to 
                # Viewer Integration" section of the Data Viewer User Manual (Help -> Contents)
                    el_tracker.sendMessage('!V CLEAR 128 128 128')
                    el_tracker.sendMessage('!V IMGLOAD CENTER ../../%s %i %i' % \
                        (ball,scn_width/2,scn_height/2)) 
                    
                #log the image onset time (in Display PC time) as a Trial Variable
                    imageTime = ball.tStartRefresh*1000
                    el_tracker.sendMessage('!V TRIAL_VAR imageTime %i' % imageTime)
                
                # set this variable to 1 to ensure we don't write the event message/
                # image load messages again on future frames
                    sentImageMessage = 1
            
            # abort the current trial if the tracker is no longer recording
                error = el_tracker.isRecording()
                if error is not pylink.TRIAL_OK:
                    el_tracker.sendMessage('tracker_disconnected')
                    abort_trial(win)
            
            # check keyboard events and then check to see if abort key combination (Ctrl-C) pressed
                keyPressList = kb.getKeys(keyList = ['lctrl','rctrl','c'], waitRelease = False, clear = False)
                for keyPress in keyPressList:
                    keyPressName = keyPress.name
                    if keyPressName not in keyPressNameList:
                        keyPressNameList.append(keyPress.name)
                if ('lctrl' in keyPressNameList or 'rctrl' in keyPressNameList) and 'c' in keyPressNameList:
                    el_tracker.sendMessage('terminated_by_user')
                    terminate_task(genv,edf_file,session_folder,session_identifier)
            #check for key releases
                keyReleaseList = kb.getKeys(keyList = ['lctrl','rctrl','c'], waitRelease = True, clear = False)
                for keyRelease in keyReleaseList:
                    keyReleaseName = keyRelease.name
                    if keyReleaseName in keyPressNameList:
                        keyPressNameList.remove(keyReleaseName)
            
            
            # *resp* updates
            waitOnFlip = False
            
            # if resp is starting this frame...
            if resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                resp.frameNStart = frameN  # exact frame index
                resp.tStart = t  # local t and not account for scr refresh
                resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'resp.started')
                # update status
                resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if resp is stopping this frame...
            if resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > resp.tStartRefresh + trial_length-frameTolerance:
                    # keep track of stop time/frame for later
                    resp.tStop = t  # not accounting for scr refresh
                    resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'resp.stopped')
                    # update status
                    resp.status = FINISHED
                    resp.status = FINISHED
            if resp.status == STARTED and not waitOnFlip:
                theseKeys = resp.getKeys(keyList=["1"], ignoreKeys=["escape"], waitRelease=False)
                _resp_allKeys.extend(theseKeys)
                if len(_resp_allKeys):
                    resp.keys = _resp_allKeys[-1].name  # just the last key pressed
                    resp.rt = _resp_allKeys[-1].rt
                    resp.duration = _resp_allKeys[-1].duration
                    # was this correct?
                    if (resp.keys == str(corrAns)) or (resp.keys == corrAns):
                        resp.corr = 1
                    else:
                        resp.corr = 0
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.075-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + trial_length-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *ball* updates
            
            # if ball is starting this frame...
            if ball.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                ball.frameNStart = frameN  # exact frame index
                ball.tStart = t  # local t and not account for scr refresh
                ball.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ball, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ball.started')
                # update status
                ball.status = STARTED
                ball.setAutoDraw(True)
            
            # if ball is active this frame...
            if ball.status == STARTED:
                # update params
                pass
            
            # if ball is stopping this frame...
            if ball.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ball.tStartRefresh + 0.075-frameTolerance:
                    # keep track of stop time/frame for later
                    ball.tStop = t  # not accounting for scr refresh
                    ball.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ball.stopped')
                    # update status
                    ball.status = FINISHED
                    ball.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial.stopped', globalClock.getTime())
        # Run 'End Routine' code from elTrial
        if int(expInfo['tracking'])==1:
        # This End Routine tab of the elTrial component clears the screen and logs
        # an additional message to mark the end of the trial
        
        # clear the screen
            clear_screen(win,genv)
            el_tracker.sendMessage('blank_screen')
        # send a message to clear the Data Viewer screen as well
            el_tracker.sendMessage('!V CLEAR 128 128 128')
            
        # send a 'TRIAL_RESULT' message to mark the end of trial, see Data
        # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
            el_tracker.sendMessage('TRIAL_RESULT %d' % 0)
        
        # update the trial counter for the next trial
            trial_index = trial_index + 1
        
        # check responses
        if resp.keys in ['', [], None]:  # No response was made
            resp.keys = None
            # was no response the correct answer?!
            if str(corrAns).lower() == 'none':
               resp.corr = 1;  # correct non-response
            else:
               resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('resp.keys',resp.keys)
        trials.addData('resp.corr', resp.corr)
        if resp.keys != None:  # we had a response
            trials.addData('resp.rt', resp.rt)
            trials.addData('resp.duration', resp.duration)
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    # get names of stimulus parameters
    if trials.trialList in ([], [None], None):
        params = []
    else:
        params = trials.trialList[0].keys()
    # save data for this loop
    trials.saveAsExcel(filename + '.xlsx', sheetName='trials',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "eyelinkStopRecording" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('eyelinkStopRecording.started', globalClock.getTime())
    # keep track of which components have finished
    eyelinkStopRecordingComponents = []
    for thisComponent in eyelinkStopRecordingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "eyelinkStopRecording" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in eyelinkStopRecordingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "eyelinkStopRecording" ---
    for thisComponent in eyelinkStopRecordingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('eyelinkStopRecording.stopped', globalClock.getTime())
    # Run 'End Routine' code from elStopRecord
    if expInfo['tracking']==1:
    # This End Routine tab of the elStopRecord component stops eye tracker recording
    
    # stop recording; add 100 msec to catch final events before stopping
        pylink.pumpDelay(100)
        el_tracker.stopRecording()
    # the Routine "eyelinkStopRecording" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "thanks" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('thanks.started', globalClock.getTime())
    # keep track of which components have finished
    thanksComponents = [endScreen]
    for thisComponent in thanksComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thanks" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *endScreen* updates
        
        # if endScreen is starting this frame...
        if endScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endScreen.frameNStart = frameN  # exact frame index
            endScreen.tStart = t  # local t and not account for scr refresh
            endScreen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endScreen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endScreen.started')
            # update status
            endScreen.status = STARTED
            endScreen.setAutoDraw(True)
        
        # if endScreen is active this frame...
        if endScreen.status == STARTED:
            # update params
            pass
        
        # if endScreen is stopping this frame...
        if endScreen.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > endScreen.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                endScreen.tStop = t  # not accounting for scr refresh
                endScreen.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'endScreen.stopped')
                # update status
                endScreen.status = FINISHED
                endScreen.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thanksComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thanks" ---
    for thisComponent in thanksComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('thanks.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    # Run 'End Experiment' code from elConnect
    # This End Experiment tab of the elConnect component calls the 
    # terminate_task helper function to get the EDF file and close the connection
    # to the Host PC
    
    # Disconnect, download the EDF file, then terminate the task
    if expInfo['tracking']==1:
        terminate_task(genv,edf_file,session_folder,session_identifier)
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
