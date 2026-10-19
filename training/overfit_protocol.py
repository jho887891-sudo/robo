"""Small, dependency-free safeguards for R3 overfit experiments.

The 32-example smoke test must draw from the training split only; it must not
consume the held-out evaluation episodes. Generation logs must store genuinely
generated token indices rather than ground-truth labels.
"""


def select_train_samples(split, episode_lengths, n=32, start_frame=20, stride=40):
    """Choose deterministic (episode, frame) pairs from TRAIN episodes only.

    `episode_lengths` maps train episode names to frame counts. For the action
    target A[t], exclude the final two frames as in the original experiment.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(start_frame, int) or start_frame < 0:
        raise ValueError("start_frame must be a nonnegative integer")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")

    train = list(split["train"])
    val = set(split["val"])
    if len(train) != len(set(train)):
        raise ValueError("duplicate episode in training split")
    if set(train) & val:
        raise ValueError("train/val episode leakage in split")

    selected = []
    for episode in sorted(train):
        length = int(episode_lengths[episode])
        if length < 0:
            raise ValueError("negative episode length: " + episode)
        for frame in range(start_frame, max(start_frame, length - 2), stride):
            selected.append([episode, frame])
            if len(selected) == n:
                return selected
    raise ValueError("not enough training examples: selected %d of %d" %
                     (len(selected), n))


def validated_generation_indices(tokens, action_dim=14, bins=256):
    """Return a JSON-safe list of actual generated action-bin indices."""
    seq = [int(x) for x in tokens]
    if len(seq) != action_dim:
        raise ValueError("expected %d generated token indices, received %d" %
                         (action_dim, len(seq)))
    if any(not (0 <= x < bins) for x in seq):
        raise ValueError("generated action token index outside [0,%d]" % (bins - 1))
    return seq
