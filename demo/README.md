# Interactive demo

A small Gradio app that makes the world model tangible: generate a drone clip,
choose how much of it the model sees, and watch it predict the rest.

```bash
pip install gradio            # optional extra
python app.py                 # untrained smoke model (pipeline demo)
python app.py --checkpoint checkpoints/world_model/latest.pt
```

Then open http://127.0.0.1:7860.

What you can do:

- **Flight clip (seed)** - draw a fresh synthetic flight.
- **Frames the model sees** - everything after this point must be *predicted*.
- **Refinement loops** - how many shared-weight passes the recurrent predictor
  takes. More loops = more "thinking time".

The readouts (rollout quality, per-loop refinement, expected exit depth) are the
same metrics reported by `scripts/evaluate.py`, just computed live for one clip.

> AeroJEPA predicts in *latent* space, not pixels, so the demo shows how closely
> the predicted representation matches the true future -- not a reconstructed
> image. That is the point of a JEPA world model: reason about *what a scene
> means* next, cheaply, rather than rendering every pixel.
