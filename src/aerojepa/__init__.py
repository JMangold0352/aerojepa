"""AeroJEPA: recurrent video world models for embodied UAV autonomy.

AeroJEPA extends the Looped-JEPA image model (a ViT context encoder, an EMA
target encoder, and a weight-shared recurrent predictor with a learned exit
gate) into the *temporal* domain. Instead of filling in masked patches of a
single image, the predictor learns to anticipate the *latent* structure of
future or masked video frames from a moving camera -- a compact, interpretable
world model for drone perception and planning.

The parent project lives at https://github.com/JMangold0352/looped-jepa.
"""

__version__ = "0.1.0"
