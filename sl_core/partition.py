
import numpy as np
from collections import defaultdict

def main_label_partition(labels, num_clients=10, iid_rate=0.8, num_classes=10, seed=1337):
    """Return index lists per client using the 'main-label' strategy.

    Each client is assigned a main class. A fraction iid_rate of its samples
    are drawn uniformly from all classes; the remaining fraction comes from
    its main class only.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    idx_by_class = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}

    # shuffle per class
    for c in idx_by_class:
        rng.shuffle(idx_by_class[c])

    # assign main class per client
    mains = rng.integers(low=0, high=num_classes, size=num_clients)

    client_indices = [[] for _ in range(num_clients)]
    # target counts roughly equal
    total = len(labels)
    per_client = total // num_clients

    for i in range(num_clients):
        need = per_client
        k_iid = int(round(iid_rate * need))
        k_main = need - k_iid

        # IID part: draw uniformly across classes
        for _ in range(k_iid):
            c = int(rng.integers(0, num_classes))
            if not idx_by_class[c]:
                # fallback: find any non-empty
                for cc in range(num_classes):
                    if idx_by_class[cc]:
                        c = cc; break
            client_indices[i].append(idx_by_class[c].pop())

        # main-label part
        main = int(mains[i])
        for _ in range(k_main):
            if not idx_by_class[main]:
                # fallback to any other class
                for cc in range(num_classes):
                    if idx_by_class[cc]:
                        main = cc; break
            client_indices[i].append(idx_by_class[main].pop())

    # put any remainder samples (if total % num_clients) to clients round-robin
    remainder = []
    for c in range(num_classes):
        remainder.extend(idx_by_class[c])
    rng.shuffle(remainder)
    for j, idx in enumerate(remainder):
        client_indices[j % num_clients].append(idx)

    return client_indices, mains.tolist()
