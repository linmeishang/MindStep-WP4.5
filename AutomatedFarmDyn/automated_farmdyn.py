# To automate the data generation process of FarmDyn
# Author: Linmei Shang & David Schäfer
# Institute for Food and Resource Economics, Universität Bonn, Germany
# Date: 25.08.2023

#%%
import pathlib
from pathlib import Path
import os
from datetime import datetime, timedelta
import shutil
import subprocess
import sched, time
import psutil


#%%
# Define a function to run FarmDyn from batch file
def runFarmDynfromBatch(FarmDynDir, IniFile, XMLFile, BATCHDir, BATCHFile):
    # Make subdirectories
    GUIDir = os.path.join(FarmDynDir, "GUI")
    BATCHFilePath = os.path.join(BATCHDir, BATCHFile)

    # General JAVA command
    javacmdstrg = "java -Xmx1G -Xverify:none -XX:+UseParallelGC -XX:PermSize=20M -XX:MaxNewSize=32M -XX:NewSize=32M -Djava.library.path=jars -classpath jars\\gig.jar de.capri.ggig.BatchExecution"

    # Append specific files to JAVA command
    javacmdparac = "{} {} {} {}".format(javacmdstrg, IniFile, XMLFile, BATCHFilePath)

    # Create batch file
    runbat = os.path.join(GUIDir, 'runfarmdyn.bat')
    if os.path.exists(runbat):
        os.remove(runbat)

    b = [runbat[:2], 'cd {}'.format(GUIDir.replace("/", "\\\\")), "SET PATH=%PATH%;./jars", javacmdparac]
    with open(runbat, "w") as f:
        f.write("\n".join(b))

    # Execute farmdyn in batch mode
    batch_process = subprocess.Popen(runbat, shell=True) 

    
 
# Start the initial run of FarmDyn run 
FarmDynDir = r'C:\schaefer_shang\Farmdyn_v1_Scenario_SimBase3' # Replace this with your FarmDyn folder
IniFile = "dairydyn.ini"  # This should stay like this
XMLFile = "dairydyn_default.xml" # This should stay like this
BATCHDir = r'C:\schaefer_shang\Farmdyn_v1_Scenario_SimBase3\GUI'  # Replace this with your FarmDyn GUI folder
# In the GUI file, there must be a bacth file in txt format. This file defines e.g. number of experiments
BATCHFile = "batch_arable_experiment.txt" # Replace this with your batch file, which must be in the GUI folder!

# Start the initial FarmDyn run, the draws are then saved in ".../results/expFarms" of your FarmDyn folder
runFarmDynfromBatch(FarmDynDir, IniFile, XMLFile, BATCHDir, BATCHFile) 


# %%
# Define a function to automatically check if the result folder still updates. 
# If there are no updates after max_pause minutes, then the saved data will be moved to a new folder, and FarmDyn will be restarted.
def automate(max_pause):
    
    f = Path('C:/schaefer_shang/Farmdyn_v1_Scenario_SimBase3/results/expFarms') # Only replace this with your FarmDyn folder until "/results/expFarms" 
    now = datetime.now()
    last_modified = os.path.getmtime(f)
    last_modified = datetime.fromtimestamp(last_modified)
    distance = now - last_modified
    print(distance)
    
    if distance > timedelta(minutes = max_pause): 
        
        print('New start is needed.')
        # Move data
        source_dir = f
        # Leaf directory 
        directory = datetime.now().strftime("%Y%m%d%H%M") # This will create a folder named by time 
        # Parent Directories 
        parent_dir = r'C:\schaefer_shang\Farmdyn_v1_Scenario_SimBase3\results\LinMeiResults'  # Replace this with the folder to where you want to move the generated data 
        # Path 
        target_dir = os.path.join(parent_dir, directory) 
        try:   
            os.makedirs(target_dir)
        except OSError:
            print ("Creation of the directory %s failed" % target_dir)
        else:
            print ("Successfully created the directory %s" % target_dir)

        file_names = os.listdir(source_dir)    
        for file_name in file_names:
            print(file_name)
            shutil.move(os.path.join(source_dir, file_name), target_dir)
        print("Data is moved")
        
        
        # Restart FarmDyn
        print("Now start FarmDyn again")
        runFarmDynfromBatch(FarmDynDir, IniFile, XMLFile, BATCHDir, BATCHFile)
        
    else:
        print("Wait")
    
    
# Define a scheduler to automatically check update and restart FarmDyn
def automation(scheduler): 
    # schedule the next call first
    scheduler.enter(check_timer, 1, automation, (scheduler,))
    # run the scheduled function
    automate(max_pause)


# %%
# Define your own max_pause abd check_timer
max_pause = 10  # in X minutes, if the result folder ".../results/expFarms" does not update longer than X minutes, then move data to new folder and restart FarmDyn
check_timer = 300 # in seconds, every X seconds check if there are still updates in the result folder

my_scheduler = sched.scheduler(time.time, time.sleep)
my_scheduler.enter(check_timer, 1, automation, (my_scheduler,))
my_scheduler.run()


