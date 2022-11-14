import yaml


def format_metrics(metrics, formatter="{:.6}"):
    newmetrics = {}
    for key, val in metrics.items():
        newmetrics[key] = formatter.format(val)
    return newmetrics


def save_metrics(path, metrics):
    with open(path, "w") as yfile:
        yaml.dump(metrics, yfile)

        
def load_metrics(path):
    with open(path, "r") as yfile:
        string = yfile.read()
        return yaml.load(string, yaml.loader.BaseLoader)
