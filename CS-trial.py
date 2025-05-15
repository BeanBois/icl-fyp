import sim
import time
import sys

# Close any open connections
sim.simxFinish(-1)

# Connect to CoppeliaSim
clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

if clientID != -1:
    print('Connected to remote API server')
    
    # Start the simulation
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
    
    # Your code here
    
    # Stop the simulation
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    
    # Close the connection
    sim.simxFinish(clientID)
else:
    print('Failed to connect to remote API server')
    sys.exit()