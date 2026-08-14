from tqdm import tqdm


def log(msg: str) -> None:
    tqdm.write(str(msg))


def tree_bar(total: int, desc: str = "trees"):
    pbar = tqdm(total=int(total), desc=desc, leave=False, unit="tree", mininterval=0.4)

    def cb(env) -> None:
        pbar.n = env.iteration + 1
        pbar.refresh()

    cb.order = 20
    cb.before_iteration = False
    return cb, pbar
