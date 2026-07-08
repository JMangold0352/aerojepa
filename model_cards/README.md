# Model cards

Professional documentation for each AeroJEPA model variant: what it is, how it
was trained, what it is good for, and where it should not be trusted.

| Card | Summary |
| --- | --- |
| [**aerojepa_base**](aerojepa_base.md) | Video encoder + feed-forward / looped predictor, masked objective. General representation learner. |
| [**aerojepa_world_model**](aerojepa_world_model.md) | Future-frame objective, recurrent predictor, optional 6-DoF action conditioning. The planning-oriented variant. |

Performance tables in these cards are filled in once real training runs complete
(roadmap Phase 2 / 5). Until then they describe the architecture, intended use,
and how to reproduce the numbers -- not invented magnitudes.
