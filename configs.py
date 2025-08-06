
_type = 'vanilla' # | "control" | "PC_EMB" | "PC_EMB_hyperbolic"
version = 2.5

CONFIGS = {
    # AGENT CONFIGS
    'NUM_AGENT_NODES' : 4,
    'PRED_HORIZON' : 5,
    'HIDDEN_DIM' : 1024, # unused 
    'NUM_ATT_HEADS' : 4, # REPLACE NODE EBD DIM 
    'HEAD_DIM' : 16, # REPLACE NODE EMB DIM 
    'AGENT_STATE_EMB_DIM' : 16,
    'EDGE_POS_DIM': 2,
    'TRAINING_MAX_TRANSLATION' : 1000, # WIDTH^2 + HEIGHT^2 SQRT
    'MAX_ROTATION_DEG' : 20,
    'GROUPING_RADIUS' : 30,

    # DEMO CONFIGS (BOTH)
    'NUM_SAMPLED_POINTCLOUDS' : 3,
    # PSEUDO-DEMO CONFIGS
    "DEMO_MAX_LENGTH" : 6,
    'DEMO_MIN_LENGTH' : 2,
    # DEMO GIVEN CONFIGS
    'SAMPLING_RATE' : 4,



    # TRAINING CONFIGS:
    'NUM_DIFFUSION_STEPS' : 500,
    'NUM_DEMO_GIVEN' : 3,
    'NUM_STEPS_PER_EPOCH' : 100,
    'NUM_EPOCHS' : 20,
    'SAVE_MODEL' : True,
    'MODEL_FILE_PATH' :  f"instant_policy_{_type}_v{version}.pth",
    'TRAINING_MAX_TRANSLATION' : 500, # WIDTH^2 + HEIGHT^2 SQRT
    'BATCH_SIZE' : 10,

    # TESTING CONFIGS
    'MAX_INFERENCE_ITER' : 100,
    'TEST_NUM_DEMO_GIVEN' : 1,
    'TESTING_MAX_UNIT_TRANSLATION' : 500, # WIDTH^2 + HEIGHT^2 SQRT
    'STATE_CHANGE_ODDS': (0.5, 0.5),
    'FIG_FILENAME' : f'avg_loss_ip_{_type}_v{version}',
    

}

from tasks2d import LousyPacmanPseudoScreenHeight as SCREEN_HEIGHT
from tasks2d import LousyPacmanPseudoScreenWidth as SCREEN_WIDTH
from tasks2d import LousyPacmanPseudoGame as  PseudoGame
