# automate the data generation process
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
def terminate(ProcessName):
    os.system('taskkill /im ' + ProcessName)
terminate('chrome.exe')

terminate('gams.exe')


#%%
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
    #os.system(runbat)
    batch_process = subprocess.Popen(runbat, shell=True) # With this batch process start running
    # Simulate running the batch file for some time
    
    
# Start the first FarmDyn run 
# Example usage
FarmDynDir = r'C:\schaefer_shang\Farmdyn_v1_Scenario_SimBase3'
IniFile = "dairydyn.ini"
XMLFile = "dairydyn_default.xml"
BATCHDir = r'C:\schaefer_shang\Farmdyn_v1_Scenario_SimBase3\GUI'
BATCHFile = "batch_arable_experiment.txt"

runFarmDynfromBatch(FarmDynDir, IniFile, XMLFile, BATCHDir, BATCHFile)  # 10 minutes


# %%
def automate(timestep):
    
    f = Path('C:/schaefer_shang/Farmdyn_v1_Scenario_SimBase3/results/expFarms')
    now = datetime.now()
    last_modified = os.path.getmtime(f)
    last_modified = datetime.fromtimestamp(last_modified)
    distance = now - last_modified
    print(distance)
    
    if distance > timedelta(minutes = timestep):
        
        print('New start is needed.')
        # Move data
        source_dir = f
        # Leaf directory 
        directory = datetime.now().strftime("%Y%m%d%H%M")
        # Parent Directories 
        parent_dir = r'C:\schaefer_shang\Farmdyn_v1_Scenario_SimBase3\results\LinMeiResults'   
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
    
    
# %%
timestep = 10  # in minutes , if the result folder does not update longer than X minutes, then move data and restart FarmDyn

# check every 600 seconds

def do_something(scheduler): 
    # schedule the next call first
    scheduler.enter(300, 1, do_something, (scheduler,))
    # run the scheduled function
    automate(timestep)
    
my_scheduler = sched.scheduler(time.time, time.sleep)
my_scheduler.enter(300, 1, do_something, (my_scheduler,))
my_scheduler.run()

#%%


