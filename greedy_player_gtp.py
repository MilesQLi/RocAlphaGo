from AlphaGo.ai import GreedyPolicyPlayer
from AlphaGo.models.policy import CNNPolicy
from interface.gtp_wrapper import run_gtp

MODEL = '../RocAlphaGo.data/models/CNNPolicy/46plane_default.json'
WEIGHTS = '../RocAlphaGo.data/training_results/supervised/46plane_GoGoD/weights.00400.hdf5'

policy = CNNPolicy.load_model(MODEL)
policy.model.load_weights(WEIGHTS)

player = GreedyPolicyPlayer(policy)

run_gtp(player)