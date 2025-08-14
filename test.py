# testing stuff we should get to this really soon 

# for pseudogame 
from tasks2d import LousyPacmanPseudoGame as PseudoGame
import random 


for _ in range(100):
    biased = random.random() > 0.5
    game = PseudoGame(
        biased = biased
    )
    game.run()
print('test passed!')

