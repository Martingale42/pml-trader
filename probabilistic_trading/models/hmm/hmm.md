class HMMActorConfig(ActorConfig):
    instrument_id: InstrumentId
    bar_type: BarType
    n_states: int = 3
    feature_window: int = 100  # For feature calculation
    update_interval: int = 3  # Days between model updates
    
class HMMFeatures:
    # Feature calculation class
    - Calculate price returns
    - Calculate price ranges
    - Calculate return categories
    - Calculate VWAP
    - Calculate lag-return/volume
    - Feature normalization
    
class HMMModel:
    # PyMC-based HMM implementation
    - Define model structure using PyMC
    - Methods for training/updating
    - Methods for prediction
    - Methods for parameter serialization/deserialization
    
class HMMState:
    # State management class
    - Track feature history
    - Track model parameters
    - Track last update time
    - Methods for state persistence
    
class HMMActor(Actor):
    def __init__(self, config: HMMActorConfig):
        - Initialize components
        - Set up feature calculator
        - Set up PyMC model
        
    def on_start(self):
        - Load persisted state if exists
        - Subscribe to required bar data
        - Set up update timer
        
    def on_bar(self, bar: Bar):
        - Update feature calculations
        - Make predictions if enough data
        - Trigger model update if needed
        
    def on_save(self):
        - Serialize model state
        
    def on_load(self):
        - Deserialize model state