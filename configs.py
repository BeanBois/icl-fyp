from tasks2d import LousyPacmanPseudoScreenHeight as SCREEN_HEIGHT
from tasks2d import LousyPacmanPseudoScreenWidth as SCREEN_WIDTH
from tasks2d import LousyPacmanPseudoGame as  PseudoGame
from tasks2d import LousyPacmanPseudoMaxForwDist as  PSEUDO_MAX_TRANSLATION
from tasks2d import LousyPacmanPseudoMaxRot as  PSEUDO_MAX_ROTATION


_type = 'vanilla' # | "control" | "PC_EMB" | "PC_EMB_hyperbolic"

version = '4_2'
# action_mode = 'small' #| 'large'
action_mode = 'large'
geo_version = '2'

SAMPLING_RATE = 10 # Vibes, the down sampling shouldnt exceeed 10 frames 

CONFIGS = {

    # GEO ENCODER TRAINING CONFIGS
    'GEO_NUM_EPOCHS' : 200,
    'GEO_BATCH_SIZE' : 5000,
    'TRAIN_GEO_ENCODER' : False,
    'NUM_SAMPLED_POINTCLOUDS' : 8,
    'GROUPING_RADIUS' : 30,


    # AGENT CONFIGS
    'NUM_AGENT_NODES' : 4,
    'PRED_HORIZON' : 5,
    'NUM_ATT_HEADS' : 4, # REPLACE NODE EBD DIM 
    'HEAD_DIM' : 16, # REPLACE NODE EMB DIM 
    'AGENT_STATE_EMB_DIM' : 16,
    'EDGE_POS_DIM': 2,
    'TRAINING_MAX_TRANSLATION' : SAMPLING_RATE * PSEUDO_MAX_TRANSLATION, # WIDTH^2 + HEIGHT^2 SQRT 
    'MAX_ROTATION_DEG' : SAMPLING_RATE * PSEUDO_MAX_ROTATION,
    'PIXEL_PER_STEP' : PSEUDO_MAX_TRANSLATION,
    'DEGREE_PER_TURN' : PSEUDO_MAX_ROTATION,



    # PSEUDO-DEMO CONFIGS
    "MAX_NUM_WAYPOINTS" : 6,
    'MIN_NUM_WAYPOINTS' : 2,
    'DEMO_MAX_LENGTH' : 10,

    # DEMO GIVEN CONFIGS
    'SAMPLING_RATE' : SAMPLING_RATE,



    # TRAINING CONFIGS:
    'NUM_DIFFUSION_STEPS' : 500,
    'NUM_DEMO_GIVEN' : 3,
    'NUM_STEPS_PER_EPOCH' : 100,
    'NUM_EPOCHS' : 20,
    'SAVE_MODEL' : True,
    'MODEL_FILE_PATH' :  f"instant_policy_{_type}_v{version}_{action_mode}.pth",
    'BATCH_SIZE' : 10,

    # TESTING CONFIGS
    'MAX_INFERENCE_ITER' : 100,
    'TEST_NUM_DEMO_GIVEN' : 1,
    'TESTING_MAX_UNIT_TRANSLATION' : SAMPLING_RATE * PSEUDO_MAX_TRANSLATION, # WIDTH^2 + HEIGHT^2 SQRT
    'TESTING_MAX_UNIT_ROTATION' : SAMPLING_RATE * PSEUDO_MAX_ROTATION,
    'STATE_CHANGE_ODDS': (0.5, 0.5),
    'FIG_FILENAME' : f'avg_loss_ip_{_type}_v{version}',
    'MANUAL_DEMO_COLLECT' : True,
    'TEST_SAMPLING_RATE' : 5,

}


