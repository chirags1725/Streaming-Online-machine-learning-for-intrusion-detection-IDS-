"""Shared helpers: NSL-KDD loading, stream generator, model factory."""
import os, random, urllib.request
import pandas as pd
from river import tree, forest, naive_bayes, neighbors, preprocessing, compose

BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"}
COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
 "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
 "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
 "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
 "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
 "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
 "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
 "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]

def download(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    for f in FILES.values():
        p = os.path.join(data_dir, f)
        if not os.path.exists(p):
            print("Downloading", f)
            urllib.request.urlretrieve(BASE + f.replace("+", "%2B"), p)

def load(data_dir="data", seed=42):
    """Returns (train_df, test_df): one-hot encoded features + binary 'y' (0 normal, 1 attack)."""
    download(data_dir)
    dfs = {k: pd.read_csv(os.path.join(data_dir, f), names=COLS) for k, f in FILES.items()}
    n_train = len(dfs["train"])
    all_df = pd.concat([dfs["train"], dfs["test"]], ignore_index=True)
    all_df["y"] = (all_df["label"] != "normal").astype(int)
    all_df["attack_type"] = all_df["label"]
    y, lab = all_df["y"], all_df["attack_type"]
    X = pd.get_dummies(all_df.drop(columns=["label", "difficulty", "y", "attack_type"]),
                       columns=CATEGORICAL, dtype=float)   # same columns for train and test
    X["y"], X["attack_type"] = y, lab
    train, test = X.iloc[:n_train].copy(), X.iloc[n_train:].copy()
    train = train.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle train: mixes classes
    return train, test                                      # test keeps its original order

def stream(df):
    """Generator that replays a dataframe instance-by-instance, like a live feed."""
    feats = [c for c in df.columns if c not in ("y", "attack_type")]
    for row, y in zip(df[feats].to_dict("records"), df["y"]):
        yield row, int(y)

def make_models():
    scaler = lambda: preprocessing.StandardScaler()
    return {
        "Hoeffding Tree": tree.HoeffdingTreeClassifier(),
        "Hoeffding Adaptive Tree": tree.HoeffdingAdaptiveTreeClassifier(seed=1),
        "Adaptive Random Forest": forest.ARFClassifier(n_models=10, seed=1),
        "Online Naive Bayes": compose.Pipeline(scaler(), naive_bayes.GaussianNB()),
        "Streaming kNN": compose.Pipeline(scaler(), neighbors.KNNClassifier(n_neighbors=5, engine=neighbors.LazySearch(window_size=1000))),
    }
